#include "util.h"

program_options parse_arguments(int argc, char *argv[])
{
	program_options op;
	int c;

	static struct option long_options[] =
	{
		{"device",required_argument,0,'d'},
		{"format",required_argument,0,'f'},
		{"help",no_argument,0,'h'},
		{"approx",required_argument,0,'k'},
		{"infile",required_argument,0,'i'},
		{"outfile",required_argument,0,'o'},
		{"verify",no_argument,0,'v'},
		{0,0,0,0} //Terminate with null
	};

	int option_index = 0;

	while((c = getopt_long(argc,argv,"d:f:hk:i:o:v",long_options,&option_index)) != -1)
	{
		switch(c)
		{
			case 'd':
				op.device = atoi(optarg);
			break;

			case 'f':
				op.format = optarg;
			break;

			case 'h':
				std::cout << "Usage: " << argv[0] << " -i <input graph file> -f <graph format: dimacs, edgelist, edgelist_h>" << std::endl;	
			exit(0);

			case 'k':
				op.approx = atoi(optarg);
			break;

			case 'i':
				op.infile = optarg;
			break;

			case 'o':
				op.outfile = optarg;
			break;

			case 'v':
				op.verify = true;
			break;

			case '?': //Invalid argument: getopt will print the error msg itself
				
			exit(1);

			default: //Fatal error
				std::cerr << "Fatal error parsing command line arguments. Terminating." << std::endl;
			exit(1);

		}
	}

	if(op.infile == NULL || op.format == NULL)
	{
		std::cerr << "Command line error: Input graph file and format are required. Use the -i and -f switches." << std::endl;
		exit(1);
	}

	return op;
}
