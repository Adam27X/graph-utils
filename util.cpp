#include "util.h"

program_options parse_arguments(int argc, char *argv[])
{
	program_options op;
	int c;

	static struct option long_options[] =
	{
		{"device",required_argument,0,'d'},
		{"help",no_argument,0,'h'},
		{"infile",required_argument,0,'i'},
		{"verify",no_argument,0,'v'},
		{0,0,0,0} //Terminate with null
	};

	int option_index = 0;

	while((c = getopt_long(argc,argv,"d:hi:v",long_options,&option_index)) != -1)
	{
		switch(c)
		{
			case 'd':
				op.device = atoi(optarg);
			break;

			case 'h':
				std::cout << "Usage: " << argv[0] << " -i <input graph file>" << std::endl;	
			exit(0);

			case 'i':
				op.infile = optarg;
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

	if(op.infile == NULL)
	{
		std::cerr << "Command line error: Input graph file is required. Use the -i switch." << std::endl;
		exit(1);
	}

	return op;
}
