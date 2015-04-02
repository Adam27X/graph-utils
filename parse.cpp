#include "parse.h"

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void host_graph::print_offset_array()
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

void host_graph::print_edge_array()
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

void host_graph::print_from_array()
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

void host_graph::print_adjacency_list()
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

host_graph parse(const program_options &op)
{
	std::string s(op.format);

	if(s == "dimacs")
	{
		return parse_metis(op.infile);
	}
	else if(s == "edgelist")
	{
		return parse_snap(op.infile,false);
	}
	else if(s == "edgelist_h")
	{
		return parse_snap(op.infile,true);
	}
	else if(s == "market")
	{
		return parse_market(op.infile);
	}
	else
	{
		std::cerr << "Error: Unsupported file type." << std::endl;
		exit(-1);
	}
}

host_graph parse_metis(char *file)
{
	host_graph g;

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
		
		//Mimic boost::split to not have a huge dependency on boost for limited functionality
		std::string temp;
		for(std::string::iterator i=line.begin(),e=line.end();i!=e;++i)
		{
			if(((*i == '\t') || (*i == ' ')) && (!temp.empty()))
			{
				splitvec.push_back(temp);
				temp.clear();
			}
			else
			{
				temp.append(i,i+1);
			}
		}
		if(!temp.empty())
		{
			splitvec.push_back(temp);
		}

		if(firstline)
		{
			g.n = stoi(splitvec[0]);
			g.m = 2*stoi(splitvec[1]);
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
			g.C.resize(g.m);
			g.F.resize(g.m);
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

	g.directed = false;

	return g;
}

host_graph parse_market(char *file)
{
        host_graph g;

        //Get n,m
        std::ifstream metis(file,std::ifstream::in);
        std::string line;
        bool firstline = true;
	bool self_edge_warned = false;

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

                //Mimic boost::split to not have a huge dependency on boost for limited functionality
		//TODO: Throw into a helper function for better reuse
                std::string temp;
                for(std::string::iterator i=line.begin(),e=line.end();i!=e;++i)
                {
                        if(((*i == '\t') || (*i == ' ')) && (!temp.empty()))
                        {
                                splitvec.push_back(temp);
                                temp.clear();
                        }
                        else
                        {
                                temp.append(i,i+1);
                        }
                }

                if(!temp.empty())
                {
                        splitvec.push_back(temp);
                }

                if(firstline)
                {
                        if(splitvec.size() != 3)
                        {
                                std::cerr << "Syntax error: The first non-comment line should include the number of rows, columns, and nonzeros in the adjacency matrix induced by the graph." << std::endl;
                                exit(-2);
                        }

			int n = stoi(splitvec[0]);
			int m = stoi(splitvec[1]);
			int l = stoi(splitvec[2]);

			if(n != m)
			{
				std::cerr << "Error: The adjacency matrix should be n by n" << std::endl;
			}

                        g.n = n;
                        g.m = 2*l;
                        firstline = false;
                        g.R.resize(g.n+1);
                }
		else
		{
			if(splitvec.size() != 2)
			{
				std::cerr << "Syntax error: Only pattern matrices are accepted at this time." << std::endl;
				std::cout << "Splitvec for offending line: " << std::endl;
				std::copy(splitvec.begin(),splitvec.end(),std::ostream_iterator<std::string>(std::cout,"\n"));
				exit(-2);
			}

			int u = stoi(splitvec[0])-1; //Mkt format indexes at 1 but we want 0-based indexing
			int v = stoi(splitvec[1])-1;

			if((u == v) && (!self_edge_warned))
			{
				std::cerr << "Warning: Self-edge detected. (" << u << "," << v << ")" << std::endl;
				self_edge_warned = true;
			}

			//Add the edge (u,v)
			g.F.push_back(u);
			g.C.push_back(v);

			//Add the edge (v,u)
			g.F.push_back(v);
			g.C.push_back(u);
		}
	}

	//Could also store unique vertices seen and compare with g.R.size()-1, but not all vertices necessarily have an edge connected to them
	if(g.F.size() != static_cast<unsigned>(g.m))
	{
		std::cerr << "Parsing error: Mismatch in the number of edges \n";
		std::cerr << "Edges claimed from the first line: " << g.m << std::endl;
		std::cerr << "Edges parsed: " << g.F.size() << std::endl;
	}


	//Need to sort by F to align edges
	std::vector< std::pair<int,int> > edges(g.F.size());
	
	for(unsigned i=0; i<g.F.size(); ++i)
	{
		edges[i] = std::make_pair(g.F[i],g.C[i]);
	}
	std::sort(edges.begin(),edges.end());
	for(unsigned i=0; i<g.F.size(); ++i)
	{
		g.F[i] = edges[i].first;
		g.C[i] = edges[i].second;
	}

        g.R[0] = 0;
        int last_node = 0;
        for(int i=0; i<g.m; i++)
        {
                while((g.F[i] > last_node) && (last_node < (g.n+1)))
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

//TODO: Handle the case where headers exist
host_graph parse_snap(char *file, bool header)
{
	host_graph g;

	std::ifstream snap(file,std::ifstream::in);
	if(!snap.good())
	{
		std::cerr << "Error opening graph file." << std::endl;
	}

	std::string line;
	std::set<int> vertices; //Keep track of the number of unique vertices
	bool extra_info_warned = false;
	bool self_edge_warned = false;
	while(std::getline(snap,line))
	{
		if(line[0] == '#')
		{
			continue;
		}

		std::vector<std::string> splitvec;

                //Mimic boost::split to not have a huge dependency on boost for limited functionality
                std::string temp;
                for(std::string::iterator i=line.begin(),e=line.end();i!=e;++i)
                {
			if(((*i == '\t') || (*i == ' ')) && (!temp.empty()))
                        {
                                splitvec.push_back(temp);
                                temp.clear();
                        }
                        else
                        {
                                temp.append(i,i+1);
                        }
                }
                if(!temp.empty())
                {
                        splitvec.push_back(temp);
                }

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

		if((u == v) && (!self_edge_warned))
		{
			std::cerr << "Warning: Self-edge detected. (" << u << "," << v << ")" << std::endl;
			self_edge_warned = true;
		}

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
		while((g.F[i] > last_node) && (last_node < (g.n+1)))
		{
			g.R[++last_node] = i;
		}
	}

	while(last_node < g.n)
	{
		g.R[++last_node] = g.m;
	}

	g.directed = true; //FIXME: For now, only support directed SNAP graphs

	return g;
}

bool host_graph::write_edgelist_to_file(const std::string &file, bool header)
{
	std::ofstream ofs(file.c_str());
	if(ofs.good())
	{
		if(header)
		{
			ofs << n << " " << m << std::endl;
		}
		for(int i=0; i<m; i++)
		{
			ofs << F[i] << " " << C[i] << std::endl;
		}
		ofs.close();
	}
	else
	{
		std::cerr << "Error opening file " << file << " for writing." << std::endl;
		return false;
	}

	return true;
}

unsigned long long host_graph::count_degree_zero_vertices()
{
	unsigned long long count = 0;
        for(auto i=0; i<n; ++i)
        {
		if(R[i] == R[i+1])
		{
			++count;
		}
        }
	return count;
}
