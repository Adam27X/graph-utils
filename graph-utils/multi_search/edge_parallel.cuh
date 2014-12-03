#pragma once

#include <vector>
#include <cuda.h>
#include <cstdio>
#include <iostream>

#include "../../parse.h"
#include "../../device_graph.h"
#include "../../util_device.cuh"

__global__ void multi_search_edge_parallel(const int *F, const int *C, const int n, const int m, int *d, size_t pitch_d, const int start, const int end);
std::vector< std::vector<int> > multi_search_edge_parallel_setup(device_graph &g, int start, int end);
