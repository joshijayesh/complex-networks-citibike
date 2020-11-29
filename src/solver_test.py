import networkx as nx
from prettytable import PrettyTable


def calc_weight(G, s, d):
    weight = 0
    weight += G.edges[('S', s)]['weight']
    weight += G.edges[(s, d)]['weight']
    weight += G.edges[(d, 'D')]['weight']
    return weight


def solve(source_list, dest_list, d_a, d_b, C_s, C_d, e):
    G = nx.DiGraph()

    for idx, node in enumerate(source_list):
        G.add_edge("S", node, weight=d_a[idx] + C_s[idx])

    for idx, node in enumerate(dest_list):
        G.add_edge(node, "D", weight=d_b[idx] + C_d[idx])

    for i, src in enumerate(source_list):
        for j, dest in enumerate(dest_list):
            G.add_edge(src, dest, weight=e[i][j])

    try:
        shortest_path = nx.dijkstra_path(G, "S", "D", weight="weight")
    except Exception:
        import pdb;pdb.set_trace()

    if(True):
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

