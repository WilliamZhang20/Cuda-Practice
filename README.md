# NVIDIA CUDA Course

All important contents from "Getting Started with Accelerated Computing in CUDA C/C++".

## Contents:

Section 1: Introduction

Section 2: Unified Memory
- Adding 2 very large vectors
- Fast implementation of SAXPY

Section 3: Streaming & Visual Profiling
- Initializating memory with CUDA streams
- Solving the n-body problem with 4096 bodies.

## The final project

The final assignment consisted of simulating the n-body problem on a GPU with force, velocity, & position computations - but accelerating a given CPU-only code by exploiting the CUDA device.

The overall mathematical procedure was simple - first, the forces are calculated in 3D space using Newton's gravitational laws from each mass to every other mass. 

Once that is done, we use the calculated forces obtain the new positions for all masses. 

The development was iterative & profile-driven, with each version analyzed using the NSight Systems Visual Profiler to analyze bottlenecks & memory usage patterns.

In my first successful version, only the force calculations were parallelized. While this "passed" the threshold, it was inefficient. Over 10 iterations, each iteration involved a transfer between host & device.

In my second version, I delegated memory transfers to only before and after all iterations of interaction computation, the latter of which was entirely parallelized. This meant that position integration had to be made into another kernel, which took the address of the array in the GPU device as an argument.

Here is what the NSys profile looks like for the best version by far (the second):
- The green at the start and pink at the end are memory transfers.
- The big blue blocks in the middle are the force calculations - they can still be accelerated with streams.
- Between the big blue blocks, one can notice a few ticks. This is the position integration procedure. 
![image](https://github.com/user-attachments/assets/804c3d50-5cb5-47e2-b913-24e233d151a9)
