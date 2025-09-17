#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

#define CHECK_CUDA(call) do {                               \
    cudaError_t _e = (call);                                \
    if (_e != cudaSuccess) {                                \
      fprintf(stderr, "CUDA ERROR %s:%d: %s\n",             \
              __FILE__, __LINE__, cudaGetErrorString(_e));  \
      cleanup(d_p, d_grid, d_green, d_fx, d_fy, d_fz, pHost, plan); \
      exit(1);                                              \
    }                                                       \
} while(0)

#define CHECK_CUFFT(call) do {                              \
    cufftResult _r = (call);                                \
    if (_r != CUFFT_SUCCESS) {                              \
      fprintf(stderr, "CUFFT ERROR %s:%d: %d\n",            \
              __FILE__, __LINE__, _r);                      \
      cleanup(d_p, d_grid, d_green, d_fx, d_fy, d_fz, pHost, plan); \
      exit(1);                                              \
    }                                                       \
} while(0)

/*
 * Algorithm parameters / heuristics
 * Grid will be cubic Nx x Nx x Nx where Nx is power-of-two.
 * GRID_FACTOR controls grid resolution relative to particle cubic root.
 */
#define GRID_FACTOR 2.0f  // grid resolution factor relative to cbrt(N)
#define MIN_GRID 32
#define MAX_GRID 256
#define EPSILON 1e-10f  // for floating-point comparisons in Green's function

// Helper: next power of two
static int next_pow2(int v) {
    if (v <= 0) return 1; // Handle non-positive inputs
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Helper: check if a number is a power of two
static int is_power_of_two(int v) {
    return (v > 0) && ((v & (v - 1)) == 0);
}

// Cleanup function to free resources
static void cleanup(Body *d_p, cufftComplex *d_grid, cufftComplex *d_green,
                    float *d_fx, float *d_fy, float *d_fz, Body *pHost, cufftHandle plan) {
    if (d_p) cudaFree(d_p);
    if (d_grid) cudaFree(d_grid);
    if (d_green) cudaFree(d_green);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (pHost) cudaFreeHost(pHost);
    cufftDestroy(plan);
}

// Kernel: zero complex grid (real part used for density/potential, imag for k-space)
__global__ void kernel_zero_grid(cufftComplex *grid, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grid[idx].x = 0.0f;
        grid[idx].y = 0.0f;
    }
}

// Kernel: scale complex grid by a factor (e.g., for FFT normalization)
__global__ void kernel_scale_complex(cufftComplex *data, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx].x *= scale;
    data[idx].y *= scale;
}

// Kernel: deposit mass to grid using Cloud-In-Cell (CIC)
__global__ void kernel_deposit_CIC(const Body *p, int n, cufftComplex *grid,
                                   int Nx, int Ny, int Nz,
                                   float xmin, float ymin, float zmin,
                                   float inv_dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = p[i].x;
    float y = p[i].y;
    float z = p[i].z;

    // Map to grid coords (floating)
    float gx = (x - xmin) * inv_dx; // in [0, Nx)
    float gy = (y - ymin) * inv_dx;
    float gz = (z - zmin) * inv_dx;

    int ix = floorf(gx);
    int iy = floorf(gy);
    int iz = floorf(gz);

    float dx = gx - ix;
    float dy = gy - iy;
    float dz = gz - iz;

    // Wrap indices for periodic boundaries
    int ix0 = (ix     ) & (Nx - 1);
    int iy0 = (iy     ) & (Ny - 1);
    int iz0 = (iz     ) & (Nz - 1);
    int ix1 = (ix + 1) & (Nx - 1);
    int iy1 = (iy + 1) & (Ny - 1);
    int iz1 = (iz + 1) & (Nz - 1);

    // Weights
    float w000 = (1-dx)*(1-dy)*(1-dz);
    float w100 = dx*(1-dy)*(1-dz);
    float w010 = (1-dx)*dy*(1-dz);
    float w001 = (1-dx)*(1-dy)*dz;
    float w110 = dx*dy*(1-dz);
    float w101 = dx*(1-dy)*dz;
    float w011 = (1-dx)*dy*dz;
    float w111 = dx*dy*dz;

    // Flatten index function
    #define IDX(ix,iy,iz) ((ix) + (iy)*Nx + (iz)*Nx*Ny)

    // Atomic add to real part (density)
    atomicAdd(&grid[IDX(ix0,iy0,iz0)].x, w000);
    atomicAdd(&grid[IDX(ix1,iy0,iz0)].x, w100);
    atomicAdd(&grid[IDX(ix0,iy1,iz0)].x, w010);
    atomicAdd(&grid[IDX(ix0,iy0,iz1)].x, w001);
    atomicAdd(&grid[IDX(ix1,iy1,iz0)].x, w110);
    atomicAdd(&grid[IDX(ix1,iy0,iz1)].x, w101);
    atomicAdd(&grid[IDX(ix0,iy1,iz1)].x, w011);
    atomicAdd(&grid[IDX(ix1,iy1,iz1)].x, w111);

    #undef IDX
}

// Kernel: multiply in k-space by Green's function (complex * complex)
__global__ void kernel_multiply_green(cufftComplex *grid_k, const cufftComplex *green_k, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    cufftComplex a = grid_k[idx];
    cufftComplex g = green_k[idx];
    // Complex multiply: (a.x + i a.y)*(g.x + i g.y)
    cufftComplex out;
    out.x = a.x * g.x - a.y * g.y;
    out.y = a.x * g.y + a.y * g.x;
    grid_k[idx] = out;
}

// Kernel: compute forces on grid by central differences on real potential grid
__global__ void kernel_compute_force_from_potential(const cufftComplex *pot, // real stored in .x
                                                   float *fx, float *fy, float *fz,
                                                   int Nx, int Ny, int Nz,
                                                   float inv_dx)
{
    int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nx*Ny*Nz;
    if (idx3 >= N) return;

    int ix = idx3 % Nx;
    int iy = (idx3 / Nx) % Ny;
    int iz = idx3 / (Nx*Ny);

    // Periodic neighbor indices
    int ixm = (ix - 1 + Nx) & (Nx - 1);
    int ixp = (ix + 1) & (Nx - 1);
    int iym = (iy - 1 + Ny) & (Ny - 1);
    int iyp = (iy + 1) & (Ny - 1);
    int izm = (iz - 1 + Nz) & (Nz - 1);
    int izp = (iz + 1) & (Nz - 1);

    int idx_xm = ixm + iy*Nx + iz*Nx*Ny;
    int idx_xp = ixp + iy*Nx + iz*Nx*Ny;
    int idx_ym = ix + iym*Nx + iz*Nx*Ny;
    int idx_yp = ix + iyp*Nx + iz*Nx*Ny;
    int idx_zm = ix + iy*Nx + izm*Nx*Ny;
    int idx_zp = ix + iy*Nx + izp*Nx*Ny;

    // Potential is stored in .x
    float phi_xm = pot[idx_xm].x;
    float phi_xp = pot[idx_xp].x;
    float phi_ym = pot[idx_ym].x;
    float phi_yp = pot[idx_yp].x;
    float phi_zm = pot[idx_zm].x;
    float phi_zp = pot[idx_zp].x;

    // Gradient: central difference
    float gx = (phi_xp - phi_xm) * 0.5f * inv_dx;
    float gy = (phi_yp - phi_ym) * 0.5f * inv_dx;
    float gz = (phi_zp - phi_zm) * 0.5f * inv_dx;

    // Force is negative gradient
    fx[idx3] = -gx;
    fy[idx3] = -gy;
    fz[idx3] = -gz;
}

// Kernel: Interpolate force to particles using CIC gather and update velocities & positions
__global__ void kernel_apply_force_CIC(Body *p, int n,
                                       const float *fx, const float *fy, const float *fz,
                                       int Nx, int Ny, int Nz,
                                       float xmin, float ymin, float zmin,
                                       float inv_dx, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = p[i].x;
    float y = p[i].y;
    float z = p[i].z;

    float gx = (x - xmin) * inv_dx; // in [0, Nx)
    float gy = (y - ymin) * inv_dx;
    float gz = (z - zmin) * inv_dx;

    int ix = floorf(gx);
    int iy = floorf(gy);
    int iz = floorf(gz);

    float dx = gx - ix;
    float dy = gy - iy;
    float dz = gz - iz;

    int ix0 = (ix     ) & (Nx - 1);
    int iy0 = (iy     ) & (Ny - 1);
    int iz0 = (iz     ) & (Nz - 1);
    int ix1 = (ix + 1) & (Nx - 1);
    int iy1 = (iy + 1) & (Ny - 1);
    int iz1 = (iz + 1) & (Nz - 1);

    #define IDX(ix,iy,iz) ((ix) + (iy)*(Nx) + (iz)*(Nx)*(Ny))

    float w000 = (1-dx)*(1-dy)*(1-dz);
    float w100 = dx*(1-dy)*(1-dz);
    float w010 = (1-dx)*dy*(1-dz);
    float w001 = (1-dx)*(1-dy)*dz;
    float w110 = dx*dy*(1-dz);
    float w101 = dx*(1-dy)*dz;
    float w011 = (1-dx)*dy*dz;
    float w111 = dx*dy*dz;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    Fx += w000 * fx[IDX(ix0,iy0,iz0)];
    Fx += w100 * fx[IDX(ix1,iy0,iz0)];
    Fx += w010 * fx[IDX(ix0,iy1,iz0)];
    Fx += w001 * fx[IDX(ix0,iy0,iz1)];
    Fx += w110 * fx[IDX(ix1,iy1,iz0)];
    Fx += w101 * fx[IDX(ix1,iy0,iz1)];
    Fx += w011 * fx[IDX(ix0,iy1,iz1)];
    Fx += w111 * fx[IDX(ix1,iy1,iz1)];

    Fy += w000 * fy[IDX(ix0,iy0,iz0)];
    Fy += w100 * fy[IDX(ix1,iy0,iz0)];
    Fy += w010 * fy[IDX(ix0,iy1,iz0)];
    Fy += w001 * fy[IDX(ix0,iy0,iz1)];
    Fy += w110 * fy[IDX(ix1,iy1,iz0)];
    Fy += w101 * fy[IDX(ix1,iy0,iz1)];
    Fy += w011 * fy[IDX(ix0,iy1,iz1)];
    Fy += w111 * fy[IDX(ix1,iy1,iz1)];

    Fz += w000 * fz[IDX(ix0,iy0,iz0)];
    Fz += w100 * fz[IDX(ix1,iy0,iz0)];
    Fz += w010 * fz[IDX(ix0,iy1,iz0)];
    Fz += w001 * fz[IDX(ix0,iy0,iz1)];
    Fz += w110 * fz[IDX(ix1,iy1,iz0)];
    Fz += w101 * fz[IDX(ix1,iy0,iz1)];
    Fz += w011 * fz[IDX(ix0,iy1,iz1)];
    Fz += w111 * fz[IDX(ix1,iy1,iz1)];

    // Update velocity: assume unit mass per particle
    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;

    // Integrate positions (simple Euler)
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;

    #undef IDX
}

// Host: precompute Green's function (k-space) for Poisson solver
// Phi_k = G_k * rho_k, G_k = -1/k^2 for k != 0, G_k = 0 for k = 0
void precompute_green(cufftComplex *green_h, int Nx, int Ny, int Nz, float Lx, float Ly, float Lz) {
    float dkx = 2.0f * M_PI / Lx;
    float dky = 2.0f * M_PI / Ly;
    float dkz = 2.0f * M_PI / Lz;

    for (int iz = 0; iz < Nz; ++iz) {
        int kz = (iz <= Nz/2) ? iz : iz - Nz;
        float kz_val = kz * dkz;
        for (int iy = 0; iy < Ny; ++iy) {
            int ky = (iy <= Ny/2) ? iy : iy - Ny;
            float ky_val = ky * dky;
            for (int ix = 0; ix < Nx; ++ix) {
                int kx = (ix <= Nx/2) ? ix : ix - Nx;
                float kx_val = kx * dkx;
                int idx = ix + iy*Nx + iz*Nx*Ny;
                float k2 = kx_val*kx_val + ky_val*ky_val + kz_val*kz_val;
                if (fabsf(k2) < EPSILON) { // Avoid division by zero
                    green_h[idx].x = 0.0f;
                    green_h[idx].y = 0.0f;
                } else {
                    // Continuum Green's function: -1/k^2
                    float val = -1.0f / k2;
                    green_h[idx].x = val;
                    green_h[idx].y = 0.0f;
                }
            }
        }
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2<<11;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);

    const char *initialized_values;
    const char *solution_values;

    if (nBodies == 2<<11) {
        initialized_values = "files/initialized_4096";
        solution_values = "solution_4096";
    } else {
        initialized_values = "files/initialized_65536";
        solution_values = "solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(Body);
    Body *pHost = NULL;
    Body *d_p = NULL;
    cufftComplex *d_grid = NULL, *d_green = NULL;
    float *d_fx = NULL, *d_fy = NULL, *d_fz = NULL;
    cufftHandle plan = 0;

    // Use pinned host memory
    CHECK_CUDA(cudaMallocHost((void**)&pHost, bytes));
    read_values_from_file(initialized_values, (float*)pHost, bytes); 

    // Compute bounding box from initial positions on host (make periodic domain)
    float xmin = pHost[0].x, ymin = pHost[0].y, zmin = pHost[0].z;
    float xmax = xmin, ymax = ymin, zmax = zmin;
    for (int i = 1; i < nBodies; ++i) {
        if (pHost[i].x < xmin) xmin = pHost[i].x;
        if (pHost[i].y < ymin) ymin = pHost[i].y;
        if (pHost[i].z < zmin) zmin = pHost[i].z;
        if (pHost[i].x > xmax) xmax = pHost[i].x;
        if (pHost[i].y > ymax) ymax = pHost[i].y;
        if (pHost[i].z > zmax) zmax = pHost[i].z;
    }
    // Tiny padding to avoid zero-size domain
    float pad = 1e-3f;
    xmin -= pad; ymin -= pad; zmin -= pad;
    xmax += pad; ymax += pad; zmax += pad;
    float Lx = xmax - xmin;
    float Ly = ymax - ymin;
    float Lz = zmax - zmin;
    if (Lx <= 0.0f) Lx = 1.0f;
    if (Ly <= 0.0f) Ly = 1.0f;
    if (Lz <= 0.0f) Lz = 1.0f;

    // Choose grid size: GRID_FACTOR * cbrt(N)
    int approx = ceilf(powf((float)nBodies, 1.0f/3.0f) * GRID_FACTOR);
    int Nx = next_pow2(approx);
    if (Nx < MIN_GRID) Nx = MIN_GRID;
    if (Nx > MAX_GRID) Nx = MAX_GRID;
    int Ny = Nx, Nz = Nx;
    int Ngrid = Nx * Ny * Nz;

    // Verify grid sizes are powers of two for periodic boundary wrapping
    assert(is_power_of_two(Nx) && is_power_of_two(Ny) && is_power_of_two(Nz));

    fprintf(stderr, "PM parameters: N=%d, grid=%d^3, L=(%g,%g,%g)\n", nBodies, Nx, Lx, Ly, Lz);

    // Allocate device particles
    CHECK_CUDA(cudaMalloc((void**)&d_p, bytes));
    CHECK_CUDA(cudaMemcpy(d_p, pHost, bytes, cudaMemcpyHostToDevice));

    // Allocate complex grid for density/potential (real in .x)
    CHECK_CUDA(cudaMalloc((void**)&d_grid, sizeof(cufftComplex) * Ngrid));
    // k-space Green's function (host -> device)
    cufftComplex *h_green = (cufftComplex*)malloc(sizeof(cufftComplex) * Ngrid);
    precompute_green(h_green, Nx, Ny, Nz, Lx, Ly, Lz);
    CHECK_CUDA(cudaMalloc((void**)&d_green, sizeof(cufftComplex) * Ngrid));
    CHECK_CUDA(cudaMemcpy(d_green, h_green, sizeof(cufftComplex) * Ngrid, cudaMemcpyHostToDevice));
    free(h_green);

    // Force grids (real floats)
    CHECK_CUDA(cudaMalloc((void**)&d_fx, sizeof(float) * Ngrid));
    CHECK_CUDA(cudaMalloc((void**)&d_fy, sizeof(float) * Ngrid));
    CHECK_CUDA(cudaMalloc((void**)&d_fz, sizeof(float) * Ngrid));

    // cuFFT plan (complex-to-complex)
    CHECK_CUFFT(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_C2C));

    // Thread config (tune TPB for specific GPU architecture if needed)
    const int TPB = 256;
    int nBlocksParticles = (nBodies + TPB - 1) / TPB;
    int nBlocksGrid = (Ngrid + TPB - 1) / TPB;

    // Precompute inverse dx for mapping positions to grid coordinates (assume cubic cells)
    float dx = Lx / (float)Nx;
    float inv_dx = 1.0f / dx;

    double totalTime = 0.0;

    for (int iter = 0; iter < nIters; ++iter) {
        StartTimer();

        // 1) Zero grid
        kernel_zero_grid<<<nBlocksGrid, TPB>>>(d_grid, Ngrid);
        CHECK_CUDA(cudaGetLastError());

        kernel_deposit_CIC<<<nBlocksParticles, TPB>>>(d_p, nBodies, d_grid, Nx, Ny, Nz,
                                                     xmin, ymin, zmin, inv_dx);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // 3) Forward FFT (grid -> k-space)
        CHECK_CUFFT(cufftExecC2C(plan, d_grid, d_grid, CUFFT_FORWARD));
        CHECK_CUDA(cudaGetLastError());

        // 4) Multiply by Green's function in k-space: Phi_k = G_k * rho_k
        kernel_multiply_green<<<nBlocksGrid, TPB>>>(d_grid, d_green, Ngrid);
        CHECK_CUDA(cudaGetLastError());

        // 5) Inverse FFT to get potential in real space (stored in .x)
        CHECK_CUFFT(cufftExecC2C(plan, d_grid, d_grid, CUFFT_INVERSE));
        CHECK_CUDA(cudaGetLastError());

        // Scale the inverse FFT output by 1/(Nx*Ny*Nz)
        float scale = 1.0f / (float)Ngrid;
        kernel_scale_complex<<<nBlocksGrid, TPB>>>(d_grid, scale, Ngrid);
        CHECK_CUDA(cudaGetLastError());

        // 6) Compute force on grid = -grad(phi) using central differences
        kernel_compute_force_from_potential<<<nBlocksGrid, TPB>>>(d_grid, d_fx, d_fy, d_fz,
                                                                 Nx, Ny, Nz, inv_dx);
        CHECK_CUDA(cudaGetLastError());

        // 7) Interpolate forces to particles and update velocities and positions
        kernel_apply_force_CIC<<<nBlocksParticles, TPB>>>(d_p, nBodies, d_fx, d_fy, d_fz,
                                                         Nx, Ny, Nz, xmin, ymin, zmin, inv_dx, dt);
        CHECK_CUDA(cudaGetLastError());

        // Sync
        CHECK_CUDA(cudaDeviceSynchronize());

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)nIters;
    // Report performance as a heuristic: N * Ngrid / avgTime
    float billionsOfOpsPerSecond = 1e-9f * nBodies * (float)Ngrid / avgTime;
    CHECK_CUDA(cudaMemcpy(pHost, d_p, bytes, cudaMemcpyDeviceToHost));

    write_values_to_file(solution_values, (float*)pHost, bytes); // Assumes files.h handles errors
    printf("%0.3f Billion 'interaction-grid' ops / second (PM heuristic)\n", billionsOfOpsPerSecond);

    // Clean up
    cleanup(d_p, d_grid, d_green, d_fx, d_fy, d_fz, pHost, plan);
    return 0;
}