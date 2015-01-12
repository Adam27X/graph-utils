#pragma once

#include <cuda.h>
#include "../util_device.cuh"

__device__ int __bfind(unsigned i);

//TODO: Templatize on predicate, forceinline?
// All threads in a warp are meant to call this function. 
// Of those threads that evaluate the predicate to be true, one will be chosen.
// This choice is actually deterministic as the largest warp_id is chosen by __bfind.
// At least one thread is expected to evaluate the predicate with the value true (could return 0 or some negative number otherwise)
// The above can be checked by a call to __any(predicate)
__device__ __forceinline__ int race_and_resolve_warp(int predicate)
{
        //Vie for control of warp
        int winner = getLaneId(); //Start with lane id

        unsigned vote = __ballot(predicate);
        if(vote)
        {
                winner = __shfl(winner,__bfind(vote));
        }

        return winner;
}
