#pragma once

#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <iterator>

#include "util.h"

//TODO: Swap in thrust::host_vector for std::vector?
class host_graph
{
public:
	void print_offset_array();
	void print_edge_array();
	void print_from_array();
	void print_adjacency_list();

	bool write_edgelist_to_file(const std::string &file, bool header); //Returns true on success, false otherwise
	unsigned long long count_degree_zero_vertices();

	//Hybrid CSR/COO representation
	std::vector<int> C; //Array of edges
	std::vector<int> R; //Array of offsets
	std::vector<int> F; //Array of where edges originate from
	int n; //Number of vertices
	int m; //Number of directed edges. For undirected graphs we treat each undirected edge as two directed edges.

	//Graph attributes
	bool directed;
};

host_graph parse(const program_options &op);
host_graph parse_metis(char *file);
host_graph parse_snap(char *file, bool header);
host_graph parse_market(char *file);
