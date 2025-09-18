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
#define M 4  // monopole + dipole x,y,z

struct Body {
    float x, y, z;
    float vx, vy, vz;
};

// Multipole: monopole + dipole
struct Multipole {
    float coeff[M];  // coeff[0]=mass, coeff[1..3]=dipole
    float cx, cy, cz; // center
};

// Leaf assignment
__global__ void assignBodiesToLeaves(int *bodyLeafMap, int nBodies, int nLeaves){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=nBodies) return;
    int leafIdx = i * nLeaves / nBodies; // simple 1D mapping
    bodyLeafMap[i] = leafIdx;
}

// Compute leaf multipoles (monopole + dipole)
__global__ void computeLeafMultipoles(Body *bodies, Multipole *leaves, int *bodyLeafMap, int nBodies){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=nBodies) return;

    int leaf = bodyLeafMap[i];
    Body b = bodies[i];

    atomicAdd(&leaves[leaf].coeff[0], 1.0f);
    atomicAdd(&leaves[leaf].coeff[1], b.x);
    atomicAdd(&leaves[leaf].coeff[2], b.y);
    atomicAdd(&leaves[leaf].coeff[3], b.z);

    atomicAdd(&leaves[leaf].cx, b.x);
    atomicAdd(&leaves[leaf].cy, b.y);
    atomicAdd(&leaves[leaf].cz, b.z);
}

// Finalize leaf centers
__global__ void finalizeLeafCenters(Multipole *leaves, int *leafCounts, int nLeaves){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=nLeaves) return;

    if(leafCounts[i]>0){
        leaves[i].cx /= leafCounts[i];
        leaves[i].cy /= leafCounts[i];
        leaves[i].cz /= leafCounts[i];
    }
}

// Compute near-field forces within leaves
__global__ void nearFieldForces(Body *bodies, int *bodyLeafMap, int nBodies, float dt){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=nBodies) return;

    int leaf = bodyLeafMap[i];
    Body bi = bodies[i];

    for(int j=0;j<nBodies;j++){
        if(i==j) continue;
        if(bodyLeafMap[j]!=leaf) continue;

        Body bj = bodies[j];
        float dx = bj.x - bi.x;
        float dy = bj.y - bi.y;
        float dz = bj.z - bi.z;
        float r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invR3 = rsqrtf(r2*r2*r2);

        bi.vx += dt * dx * invR3;
        bi.vy += dt * dy * invR3;
        bi.vz += dt * dz * invR3;
    }
    bodies[i] = bi;
}

// Apply local expansions (L2P)
__global__ void applyLocalExpansions(Body *bodies, Multipole *locals, int *bodyLeafMap, float dt, int nBodies){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=nBodies) return;

    int leaf = bodyLeafMap[i];
    Multipole local = locals[leaf];

    float dx = local.cx - bodies[i].x;
    float dy = local.cy - bodies[i].y;
    float dz = local.cz - bodies[i].z;
    float r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
    float invR3 = rsqrtf(r2*r2*r2);

    float Fx = dx * invR3 * local.coeff[0] - local.coeff[1];
    float Fy = dy * invR3 * local.coeff[0] - local.coeff[2];
    float Fz = dz * invR3 * local.coeff[0] - local.coeff[3];

    bodies[i].vx += dt * Fx;
    bodies[i].vy += dt * Fy;
    bodies[i].vz += dt * Fz;
}

// Integrate positions
__global__ void integratePositions(Body *bodies, float dt, int nBodies){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=nBodies) return;

    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
}

int main(int argc, char **argv){
    int nBodies = 2 << 11;
    if(argc>1) nBodies = 2 << atoi(argv[1]);

    Body *h_bodies = (Body*)malloc(nBodies*sizeof(Body));
    float *tmp = (float*)malloc(nBodies*6*sizeof(float));
    read_values_from_file("initialized_file", tmp, nBodies*6*sizeof(float));
    for(int i=0;i<nBodies;i++){
        h_bodies[i] = {tmp[i*6+0], tmp[i*6+1], tmp[i*6+2],
                        tmp[i*6+3], tmp[i*6+4], tmp[i*6+5]};
    }
    free(tmp);

    Body *d_bodies;
    cudaMalloc(&d_bodies, nBodies*sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, nBodies*sizeof(Body), cudaMemcpyHostToDevice);

    int nLeaves = (nBodies + MAX_BODIES_PER_LEAF -1)/MAX_BODIES_PER_LEAF;
    Multipole *d_leaves;
    int *d_bodyLeafMap;
    cudaMalloc(&d_leaves, nLeaves*sizeof(Multipole));
    cudaMalloc(&d_bodyLeafMap, nBodies*sizeof(int));
    cudaMemset(d_leaves,0,nLeaves*sizeof(Multipole));

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float dt = 0.01f;
    const int nIters = 10;
    int numBlocks = (nBodies + BLOCK_SIZE - 1)/BLOCK_SIZE;

    for(int iter=0; iter<nIters; iter++){
        StartTimer();

        assignBodiesToLeaves<<<numBlocks,BLOCK_SIZE>>>(d_bodyLeafMap,nBodies,nLeaves);

        computeLeafMultipoles<<<numBlocks,BLOCK_SIZE>>>(d_bodies,d_leaves,d_bodyLeafMap,nBodies);

        cudaMemcpy(d_leaves,d_leaves,nLeaves*sizeof(Multipole),cudaMemcpyDeviceToDevice);

        applyLocalExpansions<<<numBlocks,BLOCK_SIZE>>>(d_bodies,d_leaves,d_bodyLeafMap,dt,nBodies);

        nearFieldForces<<<numBlocks,BLOCK_SIZE>>>(d_bodies,d_bodyLeafMap,nBodies,dt);

        integratePositions<<<numBlocks,BLOCK_SIZE>>>(d_bodies,dt,nBodies);

        cudaDeviceSynchronize();
        printf("Iteration %d done.\n",iter);
    }

    cudaMemcpy(h_bodies,d_bodies,nBodies*sizeof(Body),cudaMemcpyDeviceToHost);

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
    cudaFree(d_leaves);
    cudaFree(d_bodyLeafMap);
    free(h_bodies);
    cublasDestroy(handle);

    return 0;
}
