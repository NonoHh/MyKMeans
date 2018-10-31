#include "file.h"
double *readData(const char *inputFile, int numDims, int *numPoints, int *numClusters, int *t, double *dt, int *limit, double *score, double **pointsV)
{
    FILE *f = fopen(inputFile, "r");
    if (f == NULL) {
        printf("ERROR [readData] cannot open file: %s\n", inputFile);
        return NULL;
    }
    fscanf(f, "%d %d %d %lf %d %lf\n", numPoints, numClusters, t, dt, limit, score);
    // assign the points 
    double *points = (double *)malloc((*numPoints) * numDims * sizeof(double));
    if (points == NULL) {
        printf("ERROR [readData] cannot allocate memory for points\n");
        return NULL;
    }
    *pointsV = (double *)malloc((*numPoints) * numDims * sizeof(double));
    if (*pointsV == NULL) {
        printf("ERROR [readData] cannot allocate memory for points velocity\n");
        return NULL;
    }
    // read initial points
    for (int pIndex = 0; pIndex < (*numPoints); pIndex++) {
        for (int cIndex = 0; cIndex < numDims; cIndex++) {
            fscanf(f, "%lf ", &points[cIndex + pIndex * numDims]);
        }
        for (int dIndex = 0; dIndex < numDims; dIndex++) {
            fscanf(f, "%lf ", (*pointsV) + dIndex + pIndex * numDims);
        }
        fscanf(f, "\n");
    }
    fclose(f);
    return points;
}

void writeData(char *outputFile, double **clusters, int numClusters, int numDims, double score)
{
    FILE *f = fopen(outputFile, "w");
    if (f == NULL) {
        printf("ERROR [readData] cannot open file: %s\n", outputFile);
        return;
    }
    fprintf(f, "Number of clusters with the best measure:\n\n");
    fprintf(f, "K = %d QM = %.5f\n\n", numClusters, score);
    fprintf(f, "Clusters:\n\n");
    for (int pIndex = 0; pIndex < numClusters; pIndex++) {
        fprintf(f, "%d ", pIndex + 1);
        for (int dIndex = 0; dIndex < numDims; dIndex++) {
            fprintf(f, "%.4f ", clusters[pIndex][dIndex]);
        }
        fprintf(f, "\n\n");
    }
    fclose(f);
}
