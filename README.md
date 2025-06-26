# CUDA Practice

A collection of exercises learning & practicing parallel algorithms in CUDA.

Implemented in the course "Getting Started with Accelerated Computing in CUDA C/C++".

Contains a final project to engineer a GPU-accelerated simulation of gravitationally-induced motion between many bodies, aka the N-Body Problem.

In that project, I applied both concepts from the course, as well as more advanced concepts such as coalesced memory access and shared memory.

For more details, see the section below on [the final project](#the-final-project).

## Contents:

Section 1: Introduction to CUDA
- Heat Conduction
- Matrix Multiplication
- Adding two vectors with strides

Section 2: Unified Memory
- Adding 2 very large vectors
- Fast implementation of SAXPY

Section 3: Streaming & Visual Profiling
- Initializing memory with CUDA streams
- Overlapping memory transfer & computing with CUDA streams (`cudaMemcpyAsync`)
- Solving the n-body problem with 4096 & 65536 bodies.

## The Final Project

The final assignment consisted of simulating the n-body problem on a GPU with force, velocity, & position computations - but accelerating a given CPU-only code by exploiting the CUDA device.

The overall mathematical procedure was simple - first, the forces are calculated in 3D space using Newton's gravitational laws from each mass to every other mass. 

Once that is done, we use the calculated forces to obtain the new positions for all masses. 

The development was iterative & profile-driven, with each version analyzed using the NSight Systems Visual Profiler to analyze bottlenecks & memory usage patterns.

In my first successful version, only the force calculations were parallelized. While this "passed" the threshold, it was inefficient. Over 10 iterations, each iteration involved a transfer between host & device.

In my second version, I first parallelized the position integration from the velocity amongst all bodies to the GPU, reducing the amount of serial CPU computation. Then, I also delegated memory transfers only before and after all iterations of interaction computation, since it was no longer needed on the CPU in the middle. So there was a new kernel, dealing with position integration. 

In my third version, which is in `Section_3/01-nbody.cu`, I allocated host-pinned memory for the Body objects, and removed the `cudaDeviceSynchronize()` between force & position kernels. The former was to reduce the need to copy from CPU pageable memory to the page-locked buffer accessible to the GPU's DMA by allocating directly in the pinned memory block. The latter was to remove the unnecessary synchronization, which proved to be more of a bottleneck as the kernels are synchronous.

In my fourth version, which is in `Section_3/nbody_coalesced.cu`, I implemented coalesced memory access using a parallel array, a.k.a. a Structure of Arrays (SoA), rather than the previous Array of Structures. This ensures that memory access over a consecutive memory space in the threads of a single warp is done in one clock cycle, rather than one at a time.

In my fifth version, which is in `Section_3/nbody_sharedMem.cu`, I leveraged GPU on-chip shared memory, which is accessible to all threads within a single thread block, where memory latency is much, much lower. By conducting heavy computations through data in shared memory, and then loading it into global memory at the end of a kernel, I made a huge efficiency improvement, as mentioned in the section below.

A future version would involve implementing the Barnes-Hut algorithm, involving the quadtree data structure. This would enhance the time complexity of the algorithm from an $$O(n^2)$$ to an $$O(N log(n))$$. Quite a non-trivial task, to say the least.

A simpler approach could also be to delegate things to separate streams, and then overlap computation and memory transfer using the streams. I can already see many improvements I can make to `Section_3/nbody_streams.cu`, but it is a matter of getting time to do it...

### Future Additions

The next step to accelerate the computation of the n-body problem solver is to improve its time complexity.

Currently it is O(n^2), but with the Barnes-Hut Algorithm using octrees in 3D, I can improve it to O(N*logN).

There have already been studies on it, such as [this](https://medium.com/@hsinhungw/optimizing-n-body-simulation-with-barnes-hut-algorithm-and-cuda-c76e78228c28) one using quadtrees.

## CUDA N-Body Problem Benchmarks

Performance measurements for 65536 bodies over 10 time iterations $$dt$$:
- The third version took 0.1911 seconds at 419.566 Billion I)nteractions / second
- The fourth version took 0.1813s seconds at 467.296 Billion Interactions / second
- The fifth version took 0.1443s seconds at 715.983 Billion Interactions / second

Here is what the NSys profile looks like for the 3rd version as described above:
- The green at the start and pink at the end are memory transfers.
- The big blue blocks in the middle are the force calculations - they can still be accelerated with streams.
- Between the big blue blocks, one can notice a few ticks. This is the position integration procedure. 

![image](https://github.com/user-attachments/assets/804c3d50-5cb5-47e2-b913-24e233d151a9)
