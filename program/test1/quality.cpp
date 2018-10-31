
#include "quality.h"
#include <stdio.h>

double distanceScore(int numDims, double *point1, double *point2)
{
    double dist = 0;
    for (int i = 0; i < numDims; i++) {
        dist += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(dist);
}

double* getClustersDiameters(double *points, int numPoints, int numClusters, int numDims, int *pointsBelong)
{
    int pTmpIndex, tid, threadOffset;
    int numThreads = omp_get_max_threads();
    double *diametersThreads = (double*)calloc(numThreads * numClusters, sizeof(double));
    double *diameters = (double*)malloc(numClusters * sizeof(double));
    double dist = 0;
#pragma omp parallel for private(pTmpIndex, tid, dist, threadOffset) shared(diametersThreads)
    for (int pIndex = 0; pIndex < numPoints; pIndex++) {
        int tid = omp_get_thread_num();
        threadOffset = tid * numClusters;
        for (pTmpIndex = pIndex + 1; pTmpIndex < numPoints; pTmpIndex++) {
            if (pointsBelong[pIndex] == pointsBelong[pTmpIndex]) {
                dist = distanceScore(numDims, points + (pIndex * numDims), points + (pTmpIndex * numDims));
                if (dist > diametersThreads[threadOffset + pointsBelong[pIndex]])
                    diametersThreads[threadOffset + pointsBelong[pIndex]] = dist;
            }
        }
    }
    for (int i = 0; i < numClusters; i++) {
        diameters[i] = diametersThreads[i];
        for (pTmpIndex = 1; pTmpIndex < numThreads; pTmpIndex++) {
            if (diameters[i] < diametersThreads[pTmpIndex * numClusters + i]) {
                diameters[i] = diametersThreads[pTmpIndex * numClusters + i];
            }
        }
    }
    free(diametersThreads);
    return diameters;
}

double getScore(double **clusters, int numClusters, int numDims, double *diameters)
{
    int counter;
    double score = 0;
    if (numClusters == 1) {
        counter = 1;
    }
    else {
        counter = numClusters * (numClusters - 1);
    }
    int cTmpIndex = 0;
#pragma omp parallel for private(cTmpIndex) reduction(+ : score)
    for (int cIndex = 0; cIndex < numClusters; cIndex++) {
        for (cTmpIndex = cIndex + 1; cTmpIndex < numClusters; cTmpIndex++) {
            double distance = distanceScore(numDims, clusters[cIndex], clusters[cTmpIndex]);
            if (distance != 0) {
                score += (diameters[cIndex] + diameters[cTmpIndex]);
            }
        }
    }
    return score / counter;
}
