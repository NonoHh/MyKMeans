#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double *readData(const char *inputFile, int numDims, int *numPoints, int *numClusters, int *t, double *dt, int *limit, double *score, double **pointsV);

void writeData(char *outputFile, double **clusters, int numClusters, int numDims, double score);
