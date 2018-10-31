#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void getDistance(double *cudaPoints, double *cudaClusters, int numPoints, int numClusters, int numThreadsInBlock, int numDims, double *pointsDistance);

__global__ void getMinDistance(int numPoints, int numClusters, int numThreadsInBlock, double *pointsDistance, int   *pointsBelong);

__global__ void movePoints(double *cudaPoints, double *cudaVelocity, int numPoints, int numDims, int numThreadsInBlock, double dt);

cudaError_t cudaMovePoints(double **points, double *cudaPoints, double *cudaVelocity, int numPoints, int numDims, double dt);

cudaError_t setPointsBelong(double *cudaPoints, double **clusters, int numPoints, int numClusters, int numDims, int *pointsBelong);

void cudaFreeAll(double *cudaClusters, double *pointsDistance, int *tmpPointsBelong);
