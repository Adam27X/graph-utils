#pragma once

#include "device_graph.h"
#include "util_device.cuh"
#include <iomanip>

//For now, use <<<# of SMs,max threads per block>>>
//Can autotune this later
size_t configure_grid(dim3 &dimGrid, dim3 &dimBlock, int start, int end);

//Transfer result to host. Use CUDA library calls to copy into a C-style array and then move that to a vector for convenience.
void transfer_result(const device_graph &g, int *d_d, size_t pitch_d, size_t sources_to_store, std::vector< std::vector<int> > &d_host_vector);
