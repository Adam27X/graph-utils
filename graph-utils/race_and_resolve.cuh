#pragma once

#include <cuda.h>

__device__ int __bfind(unsigned i);

//TODO: Templatize on predicate, forceinline?
// All threads in a warp are meant to call this function. 
// Of those threads that evaluate the predicate to be true, one will be chosen.
// This choice is actually deterministic as the largest warp_id is chosen by __bfind.
// At least one thread is expected to evaluate the predicate with the value true
// The above can be checked by a call to __any(predicate)
// Finally, this assumes threads are in the 'x-direction' through use of threadIdx.x. This can probably be improved.
__device__ __forceinline__ int race_and_resolve_warp(int predicate)
{
        //Vie for control of warp
        int winner = threadIdx.x & 0x1f; //Start with lane id

        unsigned vote = __ballot(predicate);
        if(vote)
        {
                winner = __shfl(winner,__bfind(vote));
        }

        return winner;
}
