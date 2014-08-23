#pragma once

#include <cstdlib>
#include <iostream>
#include <getopt.h>

#include "parse_metis.h"

//Command line parsing
class program_options
{
public:
	program_options() : infile(NULL) {}

	char *infile;
};

program_options parse_arguments(int argc, char *argv[]);
