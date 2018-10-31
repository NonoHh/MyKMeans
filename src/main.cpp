#include "file.h"
#include "kMeans.h"
#include "kernel.h"
#include "quality.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    char *inputFile = "input.txt", *outputFile = "output.txt";
    // initial MPI communication
    MPI_Init(&argc, &argv);
    int numProcs, thisID;
    MPI_Comm_rank(MPI_COMM_WORLD, &thisID);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // read data from input file
    double *points, *pointsVelocity;
    double scoreGoal, dt;
    double *doubleData = (double*)malloc(DOUBLE_SIZE * sizeof(double));
    int *pointsBelong;
    int k, t, numDims = 2, numPoints, limit;
    int *intData = (int*)malloc(INT_SIZE * sizeof(int));
    double t1 = MPI_Wtime();
    if (thisID == PID0) {
        points = readData(inputFile, numDims, &numPoints, &k, &t, &dt, &limit, &scoreGoal, &pointsVelocity);
        pointsBelong = (int*)malloc(numPoints * sizeof(int));
        intData[0] = numPoints;
        intData[1] = limit;
        intData[2] = k;
        intData[3] = t;
        doubleData[0] = scoreGoal;
        doubleData[1] = dt;
    }

    // tell all slaves needed data
    MPI_Bcast(doubleData, DOUBLE_SIZE, MPI_DOUBLE, PID0, MPI_COMM_WORLD);
    MPI_Bcast(intData, INT_SIZE, MPI_INT, PID0, MPI_COMM_WORLD);

    numPoints = intData[0];
    limit = intData[1];
    k = intData[2];
    t = intData[3];
    scoreGoal = doubleData[0];
    dt = doubleData[1];

    // initial sendCounter, displsScatter
    int *sendCounter = (int*)malloc(numProcs * sizeof(int));
    int *displsScatter = (int*)malloc(numProcs * sizeof(int));
    createPointAssign(numPoints, numProcs, numDims, sendCounter, displsScatter);

    // gather all points
    int *recvCounter = (int*)malloc(numProcs * sizeof(int));
    int *displsGather = (int*)malloc(numProcs * sizeof(int));
    int *numRecChangedPoints = (int*)malloc(numProcs * sizeof(int));
    int *numGatherChangedPoints = (int*)malloc(numProcs * sizeof(int));
    pointGatherAssign(recvCounter, sendCounter, displsGather, numRecChangedPoints,
        numGatherChangedPoints, numProcs, numDims);

    int numPointsOfProc = sendCounter[thisID] / numDims;
    double **pointsEachProc = (double**)malloc(numPointsOfProc * sizeof(double*));
    pointsEachProc[0] = (double*)malloc(numPointsOfProc * numDims * sizeof(double));
    for (int i = 1; i < numPointsOfProc; i++) {
        pointsEachProc[i] = pointsEachProc[i - 1] + numDims;
    }
    double **pointsVelocityEachProc = (double**)malloc(numPointsOfProc * sizeof(double*));
    pointsVelocityEachProc[0] = (double*)malloc(numPointsOfProc * numDims * sizeof(double));
    for (int i = 1; i < numPointsOfProc; i++) {
        pointsVelocityEachProc[i] = pointsVelocityEachProc[i - 1] + numDims;
    }

    // scatter points and velocity to all slaves
    MPI_Scatterv(points, sendCounter, displsScatter, MPI_DOUBLE, pointsEachProc[0], sendCounter[thisID], MPI_DOUBLE, PID0, MPI_COMM_WORLD);
    MPI_Scatterv(pointsVelocity, sendCounter, displsScatter, MPI_DOUBLE, pointsVelocityEachProc[0], sendCounter[thisID], MPI_DOUBLE, PID0, MPI_COMM_WORLD);

    double *cudaPoints, *cudaPointVelocity;
    cudaCopyPoints(pointsEachProc, &cudaPoints, pointsVelocityEachProc, &cudaPointVelocity, numPointsOfProc, numDims);

    double **clusters = (double**)malloc(k * sizeof(double*));
    clusters[0] = (double*)malloc(k * numDims * sizeof(double));
    for (int i = 1; i < k; i++) {
        clusters[i] = clusters[i - 1] + numDims;
    }

    // set initial cluster centers
    if (thisID == PID0)
        getInitailCenters(clusters, k, points, numPoints, numDims);
    // broadcast the initial centers
    MPI_Bcast(clusters[0], k * numDims, MPI_DOUBLE, PID0, MPI_COMM_WORLD);

    double curScore;
    double curT = 0;
    int *pointsBelongEachProc = (int*)malloc(numPointsOfProc * sizeof(int));
    do // do loop while (curT < t && curScore > scoreGoal)
    {
        if (curT != 0) {
            cudaMovePoints(pointsEachProc, cudaPoints, cudaPointVelocity, numPointsOfProc, numDims, dt);
        }
        KMeans(pointsEachProc, cudaPoints, numDims, numPointsOfProc, k, limit, pointsBelongEachProc, clusters, MPI_COMM_WORLD);
        // gather point belong together
        MPI_Gatherv(pointsBelongEachProc, numPointsOfProc, MPI_INT, pointsBelong,
            recvCounter, displsGather, MPI_INT, PID0, MPI_COMM_WORLD);
        // gather moved points together
        MPI_Gatherv(pointsEachProc[0], numPointsOfProc, MPI_DOUBLE, points, numRecChangedPoints,
            numGatherChangedPoints, MPI_DOUBLE, PID0, MPI_COMM_WORLD);
        // get cluster score
        if (thisID == PID0) {
            double *diameters = getClustersDiameters(points, numPoints, k, numDims, pointsBelong);
            curScore = getScore(clusters, k, numDims, diameters);
            free(diameters);
        }
        MPI_Bcast(&curScore, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (thisID == PID0) {
            printf("current t = %.2lf  current score = %.2lf\n", curT, curScore);
            fflush(stdout);
        }
        curT += dt;
    } while (curT < t && curScore > scoreGoal);

    if (thisID == PID0) {
        writeData(outputFile, clusters, k, numDims, curScore);
        // print the clusters
        FILE *f = fopen("result.txt", "w");
        for (int i = 0; i < numPoints; i++) {
            fprintf(f, "%d ", pointsBelong[i]);
            for (int j = 0; j < numDims; j++) {
                fprintf(f, "%lf ", points[i*numDims + j]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }
    double t2 = MPI_Wtime() - t1;
    cudaFreePoint(&cudaPoints, &cudaPointVelocity);
    free(clusters[0]);
    free(clusters);
    free(sendCounter);
    free(displsScatter);
    free(recvCounter);
    free(pointsBelongEachProc);
    free(pointsEachProc[0]);
    free(pointsEachProc);
    free(numRecChangedPoints);
    free(numGatherChangedPoints);
    free(displsGather);
    free(intData);
    free(doubleData);
    if (thisID == PID0) {
        free(pointsVelocity);
        free(points);
        free(pointsBelong);
        printf("\total time=%.5f\final score=%.5f\n\n", t2, curScore);
    }
    MPI_Finalize();
}
