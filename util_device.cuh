#pragma once

#include <iostream>
#include <stdexcept>
#include "util.h"

#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{   
    if (cudaSuccess != err)
    {   
        std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file " << file  << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#define WARP_SIZE 32

//Timing routines
void start_clock(cudaEvent_t &start, cudaEvent_t &end);
float end_clock(cudaEvent_t &start, cudaEvent_t &end);

void choose_device(const program_options &op);

//Returns the current thread's warp ID
__device__ __forceinline__ int getWarpId()
{
	return (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)/WARP_SIZE;
}

//Returns the current thread's lane ID
__device__ __forceinline__ int getLaneId()
{
	int lane_id;
	asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
	return lane_id;
}

// Extract a single bit at 'pos' from 'val'
__device__ __forceinline__ int getBit(int val, int pos)
{
	int ret;
	asm("bfe.u32 %0, %1, %2, 1;" : "=r"(ret) : "r"(val), "r"(pos));
	return ret;
}

// Insert a single bit into 'val' at position 'pos'
__device__ __forceinline__ unsigned setBit(unsigned val, unsigned toInsert, int pos)
{
	unsigned ret;
	asm("bfi.b32 %0, %1, %2, %3, 1;" : "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos));
	return ret;
}
