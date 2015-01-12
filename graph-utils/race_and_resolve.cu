#include "race_and_resolve.cuh"

//Returns the most significant bit of a 32 bit number
//Used for a pseudo "race and resolve" on a warp basis via __shfl()
__device__ int __bfind(unsigned i)
{
        int b;
        asm volatile("bfind.u32 %0, %1;" : "=r"(b) : "r"(i));
        return b;
}
