#include "kMeans.h"

void KMeans(double **points, double *cudaPoints, int numDims, int numPoints, int numClusters, int limit, int *belongs, double **clusters, MPI_Comm comm)
{
    for (int i = 0; i < numPoints; i++) {
        belongs[i] = -1;
    }
    int *cudaBelongs = (int*)malloc(numPoints * sizeof(int));
    if (cudaBelongs == NULL) {
        printf("ERROR [KMeans] cannot allocate memory for cuda belongs\n");
        return;
    }
    int *clusterSize = (int*)calloc(numClusters, sizeof(int));
    if (clusterSize == NULL) {
        printf("ERROR [KMeans] cannot allocate memory for cluster size\n");
        return;
    }
    int *newClusterSize = (int*)calloc(numClusters, sizeof(int));
    if (newClusterSize == NULL) {
        printf("ERROR [KMeans] cannot allocate memory for new cluster size\n");
        return;
    }
    double **newClusters = (double**)malloc(numClusters * sizeof(double*));
    if (newClusters == NULL) {
        printf("ERROR [KMeans] cannot allocate memory for new clusters\n");
        return;
    }
    newClusters[0] = (double*)calloc(numClusters * numDims, sizeof(double));
    if (newClusters[0] == NULL) {
        printf("ERROR [KMeans] cannot allocate memory for new clusters 0\n");
        return;
    }
    for (int i = 1; i < numClusters; i++) {
        newClusters[i] = numDims + newClusters[i - 1];
    }
    int loop = 0;
    do { // start iterations
        int belongChangedCounter = 0;
        setPointsBelong(cudaPoints, clusters, numPoints, numClusters, numDims, cudaBelongs);
        for (int i = 0; i < numPoints; i++) {
            // check if point changed its cluster
            if (belongs[i] != cudaBelongs[i]) {
                belongs[i] = cudaBelongs[i];
                belongChangedCounter++;
            }
            // update index of cluster that point i now belongs to 
            int index = cudaBelongs[i];
            newClusterSize[index]++;
            // update new cluster center
            for (int j = 0; j < numDims; j++) {
                newClusters[index][j] += points[i][j];
            }
        }
        int belongChangedSum = 0;
        // each process shares its belongChangedCounter with others
        MPI_Allreduce(&belongChangedCounter, &belongChangedSum, 1, MPI_INT, MPI_SUM, comm);
        // no point changed its cluster 
        if (belongChangedSum == 0) {
            break;
        }
        // sum all cluster data
        MPI_Allreduce(newClusters[0], clusters[0], numClusters * numDims, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT, MPI_SUM, comm);
        // update the clusters
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            for (int dIndex = 0; dIndex < numDims; dIndex++) {
                if (clusterSize[cIndex] > 1) {
                    clusters[cIndex][dIndex] /= clusterSize[cIndex];
                }
                newClusters[cIndex][dIndex] = 0.0; // reset for next iterations
            }
            newClusterSize[cIndex] = 0; // reset for next iterations
        }
        loop++;
    } while (limit > loop);
    free(newClusters[0]);
    free(newClusters);
    free(clusterSize);
    free(cudaBelongs);
    free(newClusterSize);
}

cudaError_t cudaCopyPoints(double **points, double **cudaPoints, double **pointsVelocity, double **cudaPointsVelocity, int numPoints, int numDims)
{
    cudaError_t cudaStatus = cudaSetDevice(0); // set GPU0 as the device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed");
    }
    cudaStatus = cudaMalloc((void**)cudaPoints, numPoints * numDims * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed");
        cudaFree(cudaPoints);
    }
    cudaStatus = cudaMalloc((void**)cudaPointsVelocity, numPoints * numDims * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed");
        cudaFree(cudaPoints);
    }
    // copy the points to GPU
    cudaStatus = cudaMemcpy(*cudaPoints, points[0], numPoints * numDims * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed");
        cudaFree(cudaPoints);
    }
    // copy the points velocity to GPU
    cudaStatus = cudaMemcpy(*cudaPointsVelocity, pointsVelocity[0], numPoints * numDims * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed");
        cudaFree(cudaPoints);
    }
    return cudaStatus;
}

cudaError_t cudaFreePoint(double **cudaPoints, double **cudaPointsVelocity)
{
    cudaError_t cudaStat = cudaFree(*cudaPoints);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaFree failed");
    }
    cudaStat = cudaFree(*cudaPointsVelocity);
    if (cudaStat != cudaSuccess) {
        fprintf(stderr, "cudaFree failed");
    }
    return cudaStat;
}

void movePointsWithOMP(double **points, double **velocity, int numPoints, int numDims, double dt)
{
    int dIndex = 0;
#pragma omp parallel for private(dIndex)
    for (int pIndex = 0; pIndex < numPoints; pIndex++) {
        for (dIndex = 0; dIndex < numDims; dIndex++) {
            points[pIndex][dIndex] += velocity[pIndex][dIndex] * dt;
        }
    }
}

void getInitailCenters(double** clusters, int k, double* points, int numPoints, int numDims)
{
    int dIndex = 0;
#pragma omp parallel for private(dIndex)
    for (int cIndex = 0; cIndex < k; cIndex++) {
        for (dIndex = 0; dIndex < numDims; dIndex++) {
            clusters[cIndex][dIndex] = points[dIndex + cIndex * numDims];
        }
    }
}

void createPointAssign(int numPoints, int numProcs, int numDims, int *sendCounts, int *displs)
{
    int *pointCounterForProc = (int*)malloc(numProcs * sizeof(int));
    int remainder = numPoints % numProcs;
    int index = 0;
    for (int i = 0; i < numProcs; i++) {
        pointCounterForProc[i] = numPoints / numProcs;
        if (remainder > 0) {
            remainder--;
            pointCounterForProc[i]++;
        }
        sendCounts[i] = pointCounterForProc[i] * numDims;
        displs[i] = index;
        index += sendCounts[i];
    }
    free(pointCounterForProc);
}

void pointGatherAssign(int* recvCounts, const int* sendCounts, int* displsGather, int* numRevChangedPoints, int* numDisChangedPoints, int numProcs, int numDims)
{
    for (int i = 0; i < numProcs; i++) {
        recvCounts[i] = sendCounts[i] / numDims;
    }
    displsGather[0] = 0;
    for (int i = 1; i < numProcs; i++) {
        displsGather[i] = displsGather[i - 1] + recvCounts[i - 1];
    }
    for (int i = 0; i < numProcs; i++) {
        numRevChangedPoints[i] = sendCounts[i];
    }
    numDisChangedPoints[0] = 0;
    for (int i = 1; i < numProcs; i++) {
        numDisChangedPoints[i] = numDisChangedPoints[i - 1] + numRevChangedPoints[i - 1];
    }
}