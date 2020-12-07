'''
:File: solver_test.py
:Author: Jayesh Joshi
:Email: jayeshjo1@utexas.edu

Solver portion of our optimization model
'''

import networkx as nx
from prettytable import PrettyTable
import params


def calc_weight(G, s, d):
    '''
    Calculate the summation of all the weights from source to destination

    :params G: optimization network
    :type G: Graph
    :params s: Source node
    :type s: str
    :params d: Drain node
    :type d: str
    '''
    weight = 0
    weight += G.edges[('S', s)]['weight']
    weight += G.edges[(s, d)]['weight']
    weight += G.edges[(d, 'D')]['weight']
    return weight


def solve(source_list, dest_list, d_a, d_b, C_s, C_d, e):
    '''
    Given all of the different weights for each node + edges on the optimization network, find the minimum cost to
    traverse from the source to the drain, finding the most optimal path.

    If verbose is enabled, this will output a nice table that shows how the final selection was made.

    :params source_list: List of source stations
    :type source_list: list
    :params dest_list: List of destination stations
    :type dest_list: list
    :params d_a: Distance from Source to each starting station
    :type d_a: list
    :params d_b: Distance from each destination station to the drain
    :type d_b: list
    :params C_s: Congestion at each starting node
    :type C_s: list
    :params C_d: Congestion at each destination node
    :type C_d: list
    :params e: Elevation from each stating node to each destination node
    :type e: 2D-list
    '''

    # Create the sub-optimization network
    G = nx.DiGraph()

    for idx, node in enumerate(source_list):
        G.add_edge("S", node, weight=d_a[idx] + C_s[idx])  # since dijkstra only has edge weights, move C to d

    for idx, node in enumerate(dest_list):
        G.add_edge(node, "D", weight=d_b[idx] + C_d[idx])

    for i, src in enumerate(source_list):
        for j, dest in enumerate(dest_list):
            G.add_edge(src, dest, weight=e[i][j])

    shortest_path = nx.dijkstra_path(G, "S", "D", weight="weight")

    if(params.VERBOSE):
        x = PrettyTable()
        x.field_names = ["a", "b", "distA cost", "conA cost", "eAB cost", "conB cost","distB cost","total cost"]
        x.add_row([source_list[0], dest_list[0], d_a[0], C_s[0], e[0][0], 
            C_d[0], d_b[0],
            calc_weight(G, source_list[0], dest_list[0])])
        idx_s = source_list.index(shortest_path[1])
        idx_d = dest_list.index(shortest_path[2])
        x.add_row([shortest_path[1], shortest_path[2], d_a[idx_s], C_s[idx_s],
            e[idx_s][idx_d], C_d[idx_d], d_b[idx_d],
            calc_weight(G, shortest_path[1], shortest_path[2])])

        print(x)
    
    return shortest_path[1], shortest_path[2]



if(__name__ == '__main__'):
    print(solve(
        [19, 20, 41, 59, 76],
        [11, 22, 4, 23, 24],
        [200, 300, 400, 350, 40],
        [400, 2000, 100, 200, 200],
        [69.5, 23.4, 40.4, 15.2, 90],
        [45, 26, 29.1, 15, 26],
        [15] * (5 * 5)
    ))

