#pragma once

#include <iostream>
#include <set>
#include <algorithm>

#include "../parse.h"
#include "reverse.h"

//This function assumes a DAG input. A stack overflow can occur otherwise.
void find_reach(graph &g, int i, std::vector< std::set<int> > &reach);

//Given a graph, return a vector of sets representing reachable vertices for each node
//The graph MUST be a directed acyclic graph for this method to work properly
std::vector< std::set<int> > get_reachability(graph &g);
