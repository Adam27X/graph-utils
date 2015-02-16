#include <cub/warp/warp_scan.cuh>

#include "kernel_utils.cuh"

#define WARP_SIZE 32

__device__ void load_balance_search_warp(int num_edges, int *scanned_edges, int *result);
__global__ void extract_edges(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges);
