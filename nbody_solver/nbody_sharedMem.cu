// Another massive improvement by leveraging on-chip shared memory
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

// Constants
#define BLOCK_SIZE 256

__global__ void bodyForceShared(float *x, float *y, float *z,
                                float *vx, float *vy, float *vz,
                                float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    float myX = x[i];
    float myY = y[i];
    float myZ = z[i];

    __shared__ float sh_x[BLOCK_SIZE];
    __shared__ float sh_y[BLOCK_SIZE];
    __shared__ float sh_z[BLOCK_SIZE];

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        int idx = tile + threadIdx.x;
        if (idx < n) {
            sh_x[threadIdx.x] = x[idx];
            sh_y[threadIdx.x] = y[idx];
            sh_z[threadIdx.x] = z[idx];
        } else {
            sh_x[threadIdx.x] = 0.0f;
            sh_y[threadIdx.x] = 0.0f;
            sh_z[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float dx = sh_x[j] - myX;
            float dy = sh_y[j] - myY;
            float dz = sh_z[j] - myZ;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    vx[i] += dt * Fx;
    vy[i] += dt * Fy;
    vz[i] += dt * Fz;
}

__global__ void posIntegrate(float *x, float *y, float *z,
                              float *vx, float *vy, float *vz,
                              float dt, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

int main(const int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char * initialized_values;
    const char * solution_values;
    if (nBodies == 2 << 11) {
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
    float *x, *y, *z, *vx, *vy, *vz;

    cudaMallocHost(&x, bytes);
    cudaMallocHost(&y, bytes);
    cudaMallocHost(&z, bytes);
    cudaMallocHost(&vx, bytes);
    cudaMallocHost(&vy, bytes);
    cudaMallocHost(&vz, bytes);

    float *tmp = (float*)malloc(nBodies * 6 * sizeof(float));
    read_values_from_file(initialized_values, tmp, nBodies * 6 * sizeof(float));

    for (int i = 0; i < nBodies; i++) {
        x[i] = tmp[i * 6 + 0];
        y[i] = tmp[i * 6 + 1];
        z[i] = tmp[i * 6 + 2];
        vx[i] = tmp[i * 6 + 3];
        vy[i] = tmp[i * 6 + 4];
        vz[i] = tmp[i * 6 + 5];
    }
    free(tmp);

    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);
    cudaMalloc(&d_vx, bytes);
    cudaMalloc(&d_vy, bytes);
    cudaMalloc(&d_vz, bytes);

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double totalTime = 0.0;
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        bodyForceShared<<<numBlocks, BLOCK_SIZE>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, nBodies);
        posIntegrate<<<numBlocks, BLOCK_SIZE>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, nBodies);

        cudaDeviceSynchronize();
        totalTime += GetTimer() / 1000.0;
    }

    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / (totalTime / nIters);
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    cudaMemcpy(x, d_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vx, d_vx, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vy, d_vy, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vz, d_vz, bytes, cudaMemcpyDeviceToHost);

    float *result = (float*)malloc(nBodies * 6 * sizeof(float));
    for (int i = 0; i < nBodies; i++) {
        result[i * 6 + 0] = x[i];
        result[i * 6 + 1] = y[i];
        result[i * 6 + 2] = z[i];
        result[i * 6 + 3] = vx[i];
        result[i * 6 + 4] = vy[i];
        result[i * 6 + 5] = vz[i];
    }
    write_values_to_file(solution_values, result, nBodies * 6 * sizeof(float));
    free(result);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFreeHost(x); cudaFreeHost(y); cudaFreeHost(z);
    cudaFreeHost(vx); cudaFreeHost(vy); cudaFreeHost(vz);

    return 0;
}
