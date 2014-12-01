#include <iostream>

#include "../../parse.h"
#include "../../util.h"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);

	host_graph g = parse(op.infile);

	std::cout << "Graph parsed." << std::endl;

	g.print_offset_array();
	g.print_edge_array();
	g.print_from_array();
	g.print_adjacency_list();	

	return 0;
}
