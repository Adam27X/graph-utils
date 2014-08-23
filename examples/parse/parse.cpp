#include <iostream>

#include "../../parse_metis.h"
#include "../../util.h"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);

	graph g = parse_metis(op.infile);

	std::cout << "Graph parsed." << std::endl;

	g.print_adjacency_list();	
}
