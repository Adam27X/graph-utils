#include <cub/warp/warp_scan.cuh>
#include <cub/block/block_scan.cuh>

#include "kernel_utils.cuh"

#define WARP_SIZE 32
const int BLOCK_SIZE = 512;
const int ITEMS_PER_THREAD = 2;

__device__ void load_balance_search_warp(const int vertex_frontier_size, int *edge_frontier_size, const int *edge_counts, int *scanned_edges, int *result);
__device__ void load_balance_search_block(const int vertex_frontier_size, int *edge_frontier_size, const int *edge_counts, int *scanned_edges, int *result);
__global__ void extract_edges_warp(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges);
__global__ void extract_edges_block(int vertex_frontier_size, int *edge_counts, int *scanned_edges, int *result, int *edges);
