#pragma once

#include <iostream>

graph reverse(graph &g)
{
	graph g_rev;
	g_rev.n = g.n; //Number of vertices and edges are the same for the reverse graph
	g_rev.m = g.m;

	//Sort the reversed graph's edgelist by F
	std::vector< std::pair<int,int> > tmp(g.m);
	for(int i=0; i<g.m; i++)
	{
		tmp[i] = std::make_pair(g.C[i],g.F[i]);
	}
	std::sort(tmp.begin(),tmp.end()); //Pair sorts with respect to its first argument, which is precisely what we want
	g_rev.F.resize(g.m);
	g_rev.C.resize(g.m);
	for(int i=0; i<g.m; i++)
	{
		g_rev.F[i] = tmp[i].first;
		g_rev.C[i] = tmp[i].second;
	}
	tmp.clear();

	//Induce R
        g_rev.R.resize(g.n+1);
        g_rev.R[0] = 0;
        int last_node = 0;
        for(int i=0; i<g.m; i++)
        {
                while(g_rev.F[i] > last_node)
                {
                        g_rev.R[++last_node] = i;
                }
        }

        while(last_node < g.n)
        {
                g_rev.R[++last_node] = g.m;
        }


	return g_rev;
}
