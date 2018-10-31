# K-Means Parallel Computing

## K-Means Alogrithm
​    K-Means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. K-Means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.



K-Means can be implemented in the following steps:

1. Read data from file
2. Calculate distances 
3. Calculate clusters
4. Move points according to the points' velocity and get the score of this iteration
5. If current score not enough and max time did not achieved : Back to step 2

## Parallel Computing Tools
**OpenMP:** 

Quality Calculations

**Cuda:**
Moving the points Calculating distances Calculating clusters Classifying point to cluster

**MPI (Message Passing Interface):**
Sharing and dividing the points between the processes.

## Test Result
**Input points 1**（100,000 points）

## ![input](https://github.com/NonoHh/MyKMeans/blob/master/test%20result/input.jpg)

##### **Output clusters 1**

## ![result](https://github.com/NonoHh/MyKMeans/blob/master/test%20result/result.jpg)

**Input points 2（547,359 points）**

## ![input2](https://github.com/NonoHh/MyKMeans/blob/master/test%20result/input2.jpg)

##### **Output clusters 2**

###### ![result2](https://github.com/NonoHh/MyKMeans/blob/master/test%20result/result2.jpg)
