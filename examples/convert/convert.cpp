#include <iostream>

#include "../../parse.h"
#include "../../util.h"

int main(int argc, char **argv)
{
	program_options op = parse_arguments(argc,argv);

	host_graph g = parse(op.infile);

	std::cout << "Graph parsed." << std::endl;
	std::cout << "Number of vertices: " << g.n << std::endl;
	std::cout << "Number of (directed) edges: " << g.m << std::endl;

	if(op.outfile)
	{
		if(g.write_edgelist_to_file(op.outfile,false))
		{
			std::cout << "Conversion successful" << std::endl;
		}
		else
		{
			std::cerr << "Error in file conversion" << std::endl;
		}
	}
	else
	{
		std::cerr << "Error: Cannot convert graph since no output file was given. Use the -o switch to declare an output file." << std::endl;
	}

	return 0;
}
