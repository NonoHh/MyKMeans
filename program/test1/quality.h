#pragma once

#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define INT_SIZE 4
#define DOUBLE_SIZE 2
#define PID0 0

double distanceScore(int numDims, double *point1, double *point2);

double* getClustersDiameters(double *points, int numPoints, int numClusters, int numDims, int *pointsBelong);

double getScore(double **clusters, int numClusters, int numDims, double *diameters);
