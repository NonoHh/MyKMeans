#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include "kernel.h"

void KMeans(double **points, double *cudaPoints, int numDims, int numPoints, int numClusters, int limit, int *belongs, double **clusters, MPI_Comm comm);

void movePointsWithOMP(double **points, double **velocity, int numPoints, int numDims, double dt);

void getInitailCenters(double** clusters, int k, double* points, int numPoints, int numDims);

void createPointAssign(int numPoints, int numProcs, int numDims, int *sendCounts, int *displs);

void pointGatherAssign(int* recvCounts, const int* sendCounts, int* displsGather, int* numRevChangedPoints, int* numDisChangedPoints, int numProcs, int numDims);

cudaError_t cudaCopyPoints(double **points, double **cudaPoints, double **pointsVelocity, double **cudaPointsVelocity, int numPoints, int numDims);

cudaError_t cudaFreePoint(double **cudaPoints, double **cudaPointsVelocity);
