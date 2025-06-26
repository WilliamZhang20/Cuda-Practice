// Improving memory bandwidth with coalesced access of global GPU memory
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

__global__
void bodyForce(float *x, float *y, float *z,
               float *vx, float *vy, float *vz,
               float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    float xi = x[i], yi = y[i], zi = z[i];

    for (int j = 0; j < n; j++) {
        float dx = x[j] - xi;
        float dy = y[j] - yi;
        float dz = z[j] - zi;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    vx[i] += dt * Fx;
    vy[i] += dt * Fy;
    vz[i] += dt * Fz;
}

__global__
void posIntegrate(float *x, float *y, float *z,
                  float *vx, float *vy, float *vz,
                  float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

typedef struct { float x, y, z, vx, vy, vz; } Body;

int main(const int argc, const char** argv) {
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else {
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f;
  const int nIters = 10;
  int bytes = nBodies * sizeof(float);

  float *buf;
  cudaMallocHost(&buf, nBodies * sizeof(Body));
  read_values_from_file(initialized_values, buf, nBodies * sizeof(Body));

  Body *pHost = (Body*)buf;

  float *xHost = (float*)malloc(bytes);
  float *yHost = (float*)malloc(bytes);
  float *zHost = (float*)malloc(bytes);
  float *vxHost = (float*)malloc(bytes);
  float *vyHost = (float*)malloc(bytes);
  float *vzHost = (float*)malloc(bytes);

  for (int i = 0; i < nBodies; i++) {
    xHost[i] = pHost[i].x;
    yHost[i] = pHost[i].y;
    zHost[i] = pHost[i].z;
    vxHost[i] = pHost[i].vx;
    vyHost[i] = pHost[i].vy;
    vzHost[i] = pHost[i].vz;
  }

  float *x, *y, *z, *vx, *vy, *vz;
  cudaMalloc(&x, bytes); cudaMemcpy(x, xHost, bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&y, bytes); cudaMemcpy(y, yHost, bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&z, bytes); cudaMemcpy(z, zHost, bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&vx, bytes); cudaMemcpy(vx, vxHost, bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&vy, bytes); cudaMemcpy(vy, vyHost, bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&vz, bytes); cudaMemcpy(vz, vzHost, bytes, cudaMemcpyHostToDevice);

  size_t threadsPerBlock = 256;
  size_t numBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

  double totalTime = 0.0;

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    bodyForce<<<numBlocks, threadsPerBlock>>>(x, y, z, vx, vy, vz, dt, nBodies);
    posIntegrate<<<numBlocks, threadsPerBlock>>>(x, y, z, vx, vy, vz, dt, nBodies);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  cudaMemcpy(xHost, x, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(yHost, y, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(zHost, z, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(vxHost, vx, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(vyHost, vy, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(vzHost, vz, bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < nBodies; i++) {
    pHost[i].x = xHost[i];
    pHost[i].y = yHost[i];
    pHost[i].z = zHost[i];
    pHost[i].vx = vxHost[i];
    pHost[i].vy = vyHost[i];
    pHost[i].vz = vzHost[i];
  }

  write_values_to_file(solution_values, buf, nBodies * sizeof(Body));

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  cudaFree(x); cudaFree(y); cudaFree(z);
  cudaFree(vx); cudaFree(vy); cudaFree(vz);
  cudaFreeHost(buf);
  free(xHost); free(yHost); free(zHost);
  free(vxHost); free(vyHost); free(vzHost);

  return 0;
}
