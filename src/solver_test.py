import networkx as nx


def solve(source_list, dest_list, d_a, d_b, C_s, C_d, e):
    G = nx.DiGraph()

    for idx, node in enumerate(source_list):
        G.add_edge("S", node, weight=d_a[idx] + C_s[idx])

    for idx, node in enumerate(dest_list):
        G.add_edge(node, "D", weight=d_b[idx] + C_d[idx])

    for i, src in enumerate(source_list):
        for j, dest in enumerate(dest_list):
            idx = (i * len(dest_list)) + j
            G.add_edge(src, dest, weight=e[idx])

    shortest_path = nx.dijkstra_path(G, "S", "D", weight="weight")

    return shortest_path



    '''
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    start_nodes = [0] * len(source_list) +\
                  list(np.array([[k] * len(dest_list) for k in source_list]).flatten()) +\
                  dest_list

    end_nodes = source_list +\
                dest_list * len(source_list) +\
                [0xffff] * len(dest_list)
                
    capacities = [1] * len(source_list) +\
                 [1] * (len(source_list) * len(dest_list)) +\
                 [1] * len(dest_list)

    costs = [int(sum(i)) for i in zip(d_a, C_s)] +\
            e +\
            [int(sum(i)) for i in zip(d_b, C_d)]


    source = 0
    sink = 0xffff
    supplies = [len(source_list)] + [0] * len(source_list) + [0] * len(dest_list) + [-5] # [-len(dest_list)]

    print((start_nodes))
    print((end_nodes))
    print((capacities))
    print((costs))
    print((supplies))

    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(int(start_nodes[i]), end_nodes[i], capacities[i], costs[i])

    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])


    re = min_cost_flow.Solve()
    for arc in range(min_cost_flow.NumArcs()):
        print(min_cost_flow.Head(arc), min_cost_flow.Tail(arc), min_cost_flow.UnitCost(arc))
    import pdb
    pdb.set_trace()
    if(min_cost_flow.Solve() == min_cost_flow.OPTIMAL):
        print("Total Cost = {}".format(min_cost_flow.OptimalCost()))
    else:
        raise RuntimeError("Unknown issue @ solver")
    '''


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

