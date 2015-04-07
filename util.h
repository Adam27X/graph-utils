#pragma once

#include <cstdlib>
#include <iostream>
#include <getopt.h>

//TODO: Add flag to specify the type of graph to make parsing more robust. This isn't really much of a burden for the user, especially if we provide a detailed help section.
// It also allows us to tell the user when certain types of graphs haven't been robustly tested for certain applications
//Command line parsing
class program_options
{
public:
	program_options() : device(-1), infile(NULL), outfile(NULL), verify(false), approx(0), format(NULL), isTesla(false), nvml(false) {}
	
	int device;
	char *infile;
	char *outfile;
	bool verify;
	int approx;
	char *format;
	bool isTesla;
	bool nvml;
};

program_options parse_arguments(int argc, char *argv[]);
