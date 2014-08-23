#include "parse.h"

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void graph::print_offset_array()
{
	std::cout << "R = [";
	for(auto i=R.begin(),e=R.end(); i!=e; ++i)
	{
		if(i == R.begin())
		{
			std::cout << *i;
		}
		else
		{
			std::cout << "," << *i;
		}
	}
	std::cout << "]" << std::endl;
}

void graph::print_edge_array()
{
	std::cout << "C = [";
	for(auto i=C.begin(),e=C.end(); i!=e; ++i)
	{
		if(i == C.begin())
		{
			std::cout << *i;
		}
		else
		{
			std::cout << "," << *i;
		}
	}
	std::cout << "]" << std::endl;
}

void graph::print_from_array()
{
	std::cout << "F = [";
	for(auto i=F.begin(),e=F.end(); i!=e; ++i)
	{
		if(i == F.begin())
		{
			std::cout << *i;
		}
		else
		{
			std::cout << "," << *i;
		}
	}
	std::cout << "]" << std::endl;
}

void graph::print_adjacency_list()
{
	std::cout << "Edge lists for each vertex: " << std::endl;

	for(int i=0; i<n; i++)
	{
		int begin = R[i];
		int end = R[i+1];
		for(int j=begin; j<end; j++)
		{
			if(j==begin)
			{
				std::cout << i << " | " << C[j];
			}
			else
			{
				std::cout << ", " << C[j];
			}
		}
		if(begin == end) //Single, unconnected node
		{
			std::cout << i << " | ";
		}
		std::cout << std::endl;
	}
}

graph parse(char *file)
{
	std::string s(file);

	if(s.find(".graph") != std::string::npos)
	{
		return parse_metis(file);
	}
	else if(s.find(".txt") != std::string::npos)
	{
		return parse_snap(file);
	}
	else
	{
		std::cerr << "Error: Unsupported file type." << std::endl;
		exit(-1);
	}
}

graph parse_metis(char *file)
{
	graph g;

	//Get n,m
	std::ifstream metis(file,std::ifstream::in);
	std::string line;
	bool firstline = true;
	int current_node = 0;
	int current_edge = 0;

	if(!metis.good())
	{
		std::cerr << "Error opening graph file." << std::endl;
		exit(-1);
	}

	while(std::getline(metis,line))
	{
		if(line[0] == '%')
		{
			continue;
		}

		std::vector<std::string> splitvec;
		boost::split(splitvec,line, boost::is_any_of(" \t"), boost::token_compress_on); //Now tokenize

		//If the last element is a space or tab itself, erase it
		if(!is_number(splitvec[splitvec.size()-1]))
		{
			splitvec.erase(splitvec.end()-1);
		}

		if(firstline)
		{
			g.n = stoi(splitvec[0]);
			g.m = stoi(splitvec[1]);
			if(splitvec.size() > 3)
			{
				std::cerr << "Error: Weighted graphs are not yet supported." << std::endl;
				exit(-2);
			}
			else if((splitvec.size() == 3) && (stoi(splitvec[2]) != 0))
			{
				std::cerr << "Error: Weighted graphs are not yet supported." << std::endl;
				exit(-2);
			}
			firstline = false;
			g.R.resize(g.n+1);
			g.C.resize(2*g.m);
			g.F.resize(2*g.m);
			g.R[0] = 0;
			current_node++;
		}
		else
		{
			//Count the number of edges that this vertex has and add that to the most recent value in R
			g.R[current_node] = splitvec.size()+g.R[current_node-1];
			for(unsigned i=0; i<splitvec.size(); i++)
			{
				//coPapersDBLP uses a space to mark the beginning of each line, so we'll account for that here
				if(!is_number(splitvec[i]))
				{
					//Need to adjust g.R
					g.R[current_node]--;
					continue;
				}
				//Store the neighbors in C
				//METIS graphs are indexed by one, but for our convenience we represent them as if
				//they were zero-indexed
				g.C[current_edge] = stoi(splitvec[i])-1; 
				g.F[current_edge] = current_node-1;
				current_edge++;
			}
			current_node++;
		}
	}

	return g;
}

graph parse_snap(char *file)
{
	graph g;

	std::ifstream snap(file,std::ifstream::in);
	if(!snap.good())
	{
		std::cerr << "Error opening graph file." << std::endl;
	}

	std::string line;
	std::set<int> vertices; //Keep track of the number of unique vertices
	while(std::getline(snap,line))
	{
		if(line[0] == '#')
		{
			continue;
		}

		std::vector<std::string> splitvec;
		boost::split(splitvec,line,boost::is_any_of(" \t"),boost::token_compress_on); //Now tokenize

		bool extra_info_warned = false;
		if((splitvec.size() > 2) && (!extra_info_warned))
		{
			std::cerr << "Warning: Ignoring extra information associated with each edge." << std::endl;
			std::cerr << "Example: " << std::endl;
			for(auto i=splitvec.begin()+2,e=splitvec.end(); i!=e; ++i)
			{
				std::cerr << *i << std::endl;
			}
			extra_info_warned = true;
		}

		int u = stoi(splitvec[0]);
		int v = stoi(splitvec[1]);

		g.F.push_back(u);
		g.C.push_back(v);
		vertices.insert(u);
		vertices.insert(v);
	}

	//Now induce R from F and C
	g.m = g.F.size();
	g.n = vertices.size();
	vertices.clear();

	g.R.resize(g.n+1);
	g.R[0] = 0;
	int last_node = 0;
	for(int i=0; i<g.m; i++)
	{
		while(g.F[i] > last_node)
		{
			g.R[++last_node] = i;
		}
	}

	while(last_node < g.n)
	{
		g.R[++last_node] = g.m;
	}

	return g;
}
