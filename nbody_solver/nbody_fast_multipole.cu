#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f
#define BLOCK_SIZE 256
#define MAX_BODIES_PER_LEAF 64
#define MAX_LEVELS 10
#define M 8

struct Body {
    float x, y, z;
    float vx, vy, vz;
};

// Simple multipole representation for a cluster
struct Multipole {
    float coeff[M];  // Multipole expansion coefficients
    float cx, cy, cz; // Center of cluster
};

// Kernel: assign bodies to clusters 
__global__ void assignBodiesToClusters(int *bodyClusterMap, int nBodies, int nClusters) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nBodies) return;

    int clusterIdx = i * nClusters / nBodies;
    bodyClusterMap[i] = clusterIdx;
}

__global__ void computeMultipoles(Body *bodies, Multipole *clusters, int *bodyClusterMap, int nBodies) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nBodies) return;

    int clusterIdx = bodyClusterMap[i];

    // Use atomic adds to compute cluster center (simplified)
    atomicAdd(&clusters[clusterIdx].cx, bodies[i].x);
    atomicAdd(&clusters[clusterIdx].cy, bodies[i].y);
    atomicAdd(&clusters[clusterIdx].cz, bodies[i].z);

    // For illustration, coeff[0] = mass = 1
    atomicAdd(&clusters[clusterIdx].coeff[0], 1.0f);
}

__global__ void finalizeClusterCenters(Multipole *clusters, int *clusterCounts, int nClusters) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nClusters) return;

    if (clusterCounts[i] > 0) {
        clusters[i].cx /= clusterCounts[i];
        clusters[i].cy /= clusterCounts[i];
        clusters[i].cz /= clusterCounts[i];
    }
}

// Multipole-to-local translation using cuBLAS tensor cores (simplified)
void multipoleToLocal(cublasHandle_t handle, Multipole *multipoles, Multipole *locals, int nClusters) {
    // For simplicity, treat multipole coeffs as matrices and do identity translation
    float alpha = 1.0f, beta = 0.0f;
    // A = multipoles coeffs, B = identity, C = locals
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 M, 1, 1,
                 &alpha,
                 multipoles, CUDA_R_32F, M,
                 nullptr, CUDA_R_32F, 1,
                 &beta,
                 locals, CUDA_R_32F, M,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Apply local expansions to compute body forces (simplified monopole)
__global__ void applyLocalExpansions(Body *bodies, Multipole *locals, int *bodyClusterMap, float dt, int nBodies) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nBodies) return;

    int clusterIdx = bodyClusterMap[i];
    Multipole local = locals[clusterIdx];

    float dx = local.cx - bodies[i].x;
    float dy = local.cy - bodies[i].y;
    float dz = local.cz - bodies[i].z;
    float r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
    float invR3 = rsqrtf(r2 * r2 * r2);

    float Fx = dx * invR3 * local.coeff[0]; // coeff[0] = total mass
    float Fy = dy * invR3 * local.coeff[0];
    float Fz = dz * invR3 * local.coeff[0];

    bodies[i].vx += dt * Fx;
    bodies[i].vy += dt * Fy;
    bodies[i].vz += dt * Fz;
}

// Integrate positions
__global__ void integratePositions(Body *bodies, float dt, int nBodies) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nBodies) return;

    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
}

int main(int argc, char **argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    Body *h_bodies = (Body*)malloc(nBodies * sizeof(Body));
    float *tmp = (float*)malloc(nBodies * 6 * sizeof(float));
    read_values_from_file("initialized_file", tmp, nBodies*6*sizeof(float));
    for(int i=0;i<nBodies;i++){
        h_bodies[i] = {tmp[i*6+0], tmp[i*6+1], tmp[i*6+2],
                        tmp[i*6+3], tmp[i*6+4], tmp[i*6+5]};
    }
    free(tmp);

    Body *d_bodies;
    cudaMalloc(&d_bodies, nBodies * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, nBodies*sizeof(Body), cudaMemcpyHostToDevice);

    int nClusters = (nBodies + MAX_BODIES_PER_LEAF - 1) / MAX_BODIES_PER_LEAF;
    Multipole *d_multipoles, *d_locals;
    int *d_bodyClusterMap;
    cudaMalloc(&d_multipoles, nClusters * sizeof(Multipole));
    cudaMalloc(&d_locals, nClusters * sizeof(Multipole));
    cudaMalloc(&d_bodyClusterMap, nBodies * sizeof(int));

    // Clear multipoles
    cudaMemset(d_multipoles, 0, nClusters*sizeof(Multipole));
    cudaMemset(d_locals, 0, nClusters*sizeof(Multipole));

    // Initialize cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float dt = 0.01f;
    const int nIters = 10;
    int numBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        assignBodiesToClusters<<<numBlocks, BLOCK_SIZE>>>(d_bodyClusterMap, nBodies, nClusters);

        computeMultipoles<<<numBlocks, BLOCK_SIZE>>>(d_bodies, d_multipoles, d_bodyClusterMap, nBodies);

        multipoleToLocal(handle, d_multipoles, d_locals, nClusters);

        applyLocalExpansions<<<numBlocks, BLOCK_SIZE>>>(d_bodies, d_locals, d_bodyClusterMap, dt, nBodies);

        integratePositions<<<numBlocks, BLOCK_SIZE>>>(d_bodies, dt, nBodies);

        cudaDeviceSynchronize();
        printf("Iteration %d done.\n", iter);
    }

    cudaMemcpy(h_bodies, d_bodies, nBodies*sizeof(Body), cudaMemcpyDeviceToHost);

    float *result = (float*)malloc(nBodies*6*sizeof(float));
    for(int i=0;i<nBodies;i++){
        result[i*6+0] = h_bodies[i].x;
        result[i*6+1] = h_bodies[i].y;
        result[i*6+2] = h_bodies[i].z;
        result[i*6+3] = h_bodies[i].vx;
        result[i*6+4] = h_bodies[i].vy;
        result[i*6+5] = h_bodies[i].vz;
    }
    write_values_to_file("solution_file", result, nBodies*6*sizeof(float));
    free(result);

    cudaFree(d_bodies);
    cudaFree(d_multipoles);
    cudaFree(d_locals);
    cudaFree(d_bodyClusterMap);
    free(h_bodies);
    cublasDestroy(handle);

    return 0;
}
