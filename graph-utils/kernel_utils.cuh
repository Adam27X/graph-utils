#pragma once

#include <cuda.h>

#define WARP_SIZE 32

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
