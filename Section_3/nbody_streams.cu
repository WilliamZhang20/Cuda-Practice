#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

__global__
void bodyForce(Body *p, float dt, int n, int startIdx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + startIdx;

    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) { // future goal - PARALLELIZE this with streams!!!
      if(i == j) continue;
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;
    
      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
}

__global__
void posIntegrate(Body *p, float dt, int n, int startIdx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x + startIdx;
    if (i >= n) return;  // Prevent out-of-bounds memory access

    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
}

int main(const int argc, const char** argv) {

  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody report files
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]); // possible reassignment of nBodies

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;
	 
  cudaHostAlloc(&buf, bytes, cudaHostAllocDefault);


  read_values_from_file(initialized_values, buf, bytes);

  Body *pHost = (Body*)buf; // CPU reinterpretation at the pointer value

  Body *p; // to be stored in GPU
  cudaMalloc(&p, bytes);

  const size_t numStreams = 4;
  cudaStream_t streams[numStreams];
  for (int i = 0; i < numStreams; ++i) {
     cudaStreamCreate(&streams[i]);
     cudaMemcpyAsync(p, pHost, bytes, cudaMemcpyHostToDevice, streams[i]);
  }
  
  size_t threadsPerBlock;
  threadsPerBlock = 256;
  // Whatever the # of threads per block, those threads aren't in parallel
  // Warps of same instruction are parallel - i.e. SIMT
  
  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
      /*
    * You will likely wish to refactor the work being done in bodyForce,
    * and potentially the work to integrate the positions.
    */

    int chunkSize = nBodies / numStreams;
    int chunkDiv = chunkSize / threadsPerBlock;
    for (int streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
        int startIdx = streamIdx * chunkSize;
        bodyForce<<<chunkDiv, threadsPerBlock, 0, streams[streamIdx]>>>(p, dt, nBodies, startIdx);
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    /*
    * This position integration cannot occur until this round of bodyForce has completed.
    * Also, the next round of bodyForce cannot begin until the integration is complete.
    */

    for (int streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
        int startIdx = streamIdx * chunkSize;
        posIntegrate<<<chunkDiv, threadsPerBlock, 0, streams[streamIdx]>>>(p, dt, nBodies, startIdx);
    }
        
    // Parallelized position integration since many operations of same style
    // since nBodies operations - same optimization mechanism for grid config
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaError_t asyncErr;
    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
    
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

  for (int i = 0; i < numStreams; ++i) {
     cudaMemcpyAsync(buf, p, bytes, cudaMemcpyDeviceToHost, streams[i]);
  }
 
  write_values_to_file(solution_values, buf, bytes);

  // You will likely enjoy watching this value grow as you accelerate the application,
  // but beware that a failure to correctly synchronize the device might result in
  // unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);


  cudaFree(p);
  cudaFreeHost(buf);
}