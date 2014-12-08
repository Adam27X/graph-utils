#pragma once

#include <cstdlib>
#include <iostream>
#include <getopt.h>

//Command line parsing
class program_options
{
public:
	program_options() : device(-1), infile(NULL) {}
	
	int device;
	char *infile;
};

program_options parse_arguments(int argc, char *argv[]);
