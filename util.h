#pragma once

#include <cstdlib>
#include <iostream>
#include <getopt.h>

//Command line parsing
class program_options
{
public:
	program_options() : device(-1), infile(NULL), outfile(NULL), verify(false), approx(0) {}
	
	int device;
	char *infile;
	char *outfile;
	bool verify;
	int approx;
};

program_options parse_arguments(int argc, char *argv[]);
