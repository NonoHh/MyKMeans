#include "kernel.h"

__global__ void getDistance(double *cudaPoints, double *cudaClusters, int numPoints, int numClusters, int numThreadsInBlock, int numDims, double *pointsDistance)
{
    double result = 0;
    int blockID = blockIdx.x;
    if ((blockID + 1 == gridDim.x) && (numPoints % blockDim.x <= threadIdx.x)) {
        return;
    }
    for (int dIndex = 0; dIndex < numDims; dIndex++) {
        result += (cudaPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + dIndex] - cudaClusters[threadIdx.y * numDims + dIndex])
            * (cudaPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + dIndex] - cudaClusters[threadIdx.y * numDims + dIndex]);
    }
    // update distance
    pointsDistance[numPoints * threadIdx.y + (blockID * numThreadsInBlock + threadIdx.x)] = result;
}

__global__ void getMinDistance(int numPoints, int numClusters, int numThreadsInBlock, double *pointsDistance, int   *pointsBelong)
{
    int blockId = blockIdx.x;
    double minIndex = 0;
    if ((blockIdx.x == gridDim.x - 1) && (numPoints % blockDim.x <= threadIdx.x)) {
        return;
    }
    double minDistance = pointsDistance[(numThreadsInBlock * blockId) + threadIdx.x];
    for (int cIndex = 1; cIndex < numClusters; cIndex++) {
        double tmpDistance = pointsDistance[(numThreadsInBlock * blockId) + threadIdx.x + (cIndex * numPoints)];
        if (minDistance > tmpDistance) {
            minDistance = tmpDistance;
            minIndex = cIndex;
        }
    }
    // update belong
    pointsBelong[numThreadsInBlock * blockId + threadIdx.x] = minIndex;
}

__global__ void movePoints(double *cudaPoints, double *cudaVelocity, int numPoints, int numDims, int numThreadsInBlock, double dt)
{
    int blockID = blockIdx.x;
    if ((blockID + 1 == gridDim.x) && (numPoints % blockDim.x <= threadIdx.x)) {
        return;
    }
    for (int dIndex = 0; dIndex < numDims; dIndex++) {
        cudaPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + dIndex] += dt * cudaVelocity[(blockID * numThreadsInBlock + threadIdx.x) * numDims + dIndex];
    }
}

cudaError_t cudaMovePoints(double **points, double *cudaPoints, double *cudaVelocity, int numPoints, int numDims, double dt)
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int numThreadsInBlock = devProp.maxThreadsPerBlock;
    int numBlocks = numPoints / numThreadsInBlock;
    if (numPoints % numThreadsInBlock > 0) {
        numBlocks++;
    }
    movePoints << <numBlocks, numThreadsInBlock >> > (cudaPoints, cudaVelocity, numPoints, numDims, numThreadsInBlock, dt);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy((void**)points[0], cudaPoints, numPoints * numDims * sizeof(double), cudaMemcpyDeviceToHost);
    return cudaStatus;
}

cudaError_t setPointsBelong(double *cudaPoints, double **clusters, int numPoints, int numClusters, int numDims, int *pointsBelong)
{
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int numThreadsInBlock = devProp.maxThreadsPerBlock / numClusters;
    dim3 dim(numThreadsInBlock, numClusters);
    int numBlocks = numPoints / numThreadsInBlock;
    if (numPoints % numThreadsInBlock > 0) {
        numBlocks++;
    }
    double *cudaClusters;
    double *pointsDistance = 0;
    int *tmpPointsBelong = 0;
    cudaError_t cudaStatus = cudaMalloc((void**)&cudaClusters, numClusters * numDims * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed");
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&pointsDistance, numClusters * numPoints * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed");
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&tmpPointsBelong, numPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed");
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(cudaClusters, clusters[0], numClusters * numDims * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed");
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    getDistance << <numBlocks, dim >> > (cudaPoints, cudaClusters, numPoints, numClusters, numThreadsInBlock, numDims, pointsDistance);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize: error code %d\n", cudaStatus);
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    numThreadsInBlock = devProp.maxThreadsPerBlock;
    numBlocks = numPoints / numThreadsInBlock;
    if (numPoints % numThreadsInBlock > 0) { numBlocks++; }
    getMinDistance << <numBlocks, numThreadsInBlock >> > (numPoints, numClusters, numThreadsInBlock, pointsDistance, tmpPointsBelong);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize: error code %d\n", cudaStatus);
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    cudaStatus = cudaMemcpy(pointsBelong, tmpPointsBelong, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed");
        cudaFreeAll(cudaClusters, pointsDistance, tmpPointsBelong);
        return cudaStatus;
    }
    return cudaStatus;
}

void cudaFreeAll(double *cudaClusters, double *pointsDistance, int *tmpPointsBelong)
{
    cudaFree(cudaClusters);
    cudaFree(pointsDistance);
    cudaFree(tmpPointsBelong);
}
