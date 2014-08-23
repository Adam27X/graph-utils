#pragma once

#include <iostream>
#include <string>
#include <set>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <cstdlib>

class graph
{
public:
	void print_offset_array();
	void print_edge_array();
	void print_from_array();
	void print_adjacency_list();

	//Hybrid CSR/COO representation
	std::vector<int> C; //Array of edges
	std::vector<int> R; //Array of offsets
	std::vector<int> F; //Array of where edges originate from
	int n; //Number of vertices
	int m; //Number of edges
};

graph parse_metis(char *file);
