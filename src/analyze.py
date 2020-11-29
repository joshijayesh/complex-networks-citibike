import click
import csv
import networkx as nx
import pandas as pd
import numpy as np
import googlemaps
import time
import collections
import random
from datetime import datetime
from pathlib import Path
from numba.typed import List
from numba import njit
from math import cos, asin, sqrt
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from pylab import rcParams
from solver_test import solve
rcParams['figure.figsize'] = 12, 12


# These are for the finding proximity
CLEAR_SURROUND = 0.0001  # This var is used to cut off a bunch of stuff from the elevation LUT
MIN_SURROUND = 5
MAX_SURROUND = 10

# These are etc
HOUR_TO_COLLECT = 17
CONGESTION_CONSIDERATION = 50.0
COMPLIANCE_RATE = 0.2

SCALE_d = 10
SCALE_e = 10

WEIGHT_S_a = (1/250)
WEIGHT_a = 1
WEIGHT_a_b = (1/1500)
WEIGHT_b = 1
WEIGHT_b_D = (1/250)

OFFSET_e = 500


@njit
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))


@njit
def cut_data(data, ll_tuple, surround):
    new_data = List()
    lon_min = ll_tuple[0] * (1 + surround)  # assuming lon is always negative
    lon_max = ll_tuple[0] * (1 - surround)
    lat_min = ll_tuple[1] * (1 - surround)
    lat_max = ll_tuple[1] * (1 + surround)

    for lon, lat in data:
        if(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            new_data.append((lon, lat, ))

    return new_data


def closest(data, ll_tuple, spike=4, surround=CLEAR_SURROUND):
    cutted_data = cut_data(data, ll_tuple, surround)
    if(not(cutted_data)):
        print("NO")
        cutted_data = cut_data(data, ll_tuple, surround * spike)

    return cutted_data


def find_proximity_graph(nodes):
    gmaps = googlemaps.Client(key="AIzaSyCQ2HFIRPomOgGRNrXSnKdltYxGGYfeGx0")
    G = nx.Graph()
    df = pd.read_csv(nodes)
    ll_dict = {}
    for index, row in df.iterrows():
        node_name = row['ID']
        latitude = row['Latitude']
        longitude = row['Longitude']

        ll_dict[(longitude, latitude)] = node_name

        G.add_node(node_name, latitude=latitude, longitude=longitude)

    ll_list = np.array(list(ll_dict.keys()))
    for ll, node in ll_dict.items():  # Maybe this needs to be better, but here I'm just trying to find some closest nodes
        closest_stations = closest(ll_list, ll)
        len_stations = len(closest_stations)
        if(len_stations < MIN_SURROUND):
            closest_stations = closest(ll_list, ll, surround=CLEAR_SURROUND * (2 if len_stations <= 3 else 1.15))
        elif(len_stations > MAX_SURROUND):
            closest_stations = closest(ll_list, ll, surround=CLEAR_SURROUND / (len_stations / MAX_SURROUND))
        
        len_stations = len(closest_stations)
        if(len_stations > (MAX_SURROUND + 5)):
            closest_stations = closest(ll_list, ll, surround=CLEAR_SURROUND / ((len_stations) / (3 * MAX_SURROUND)))

        assert len(closest_stations) > 1, "Cannot be just 1... since that indicates the only closest is itself"

        for tgt_ll in closest_stations:
            tgt_node = ll_dict[tgt_ll]
            
            origin_latitude = tgt_ll[1]
            origin_longitude = tgt_ll[0]
            dest_latitude = ll[1]
            dest_longitude = ll[0]


            if(tgt_node != node):
                dist = gmaps.distance_matrix([str(origin_latitude) + " " + str(origin_longitude)],
                                             [str(dest_latitude) + " " + str(dest_longitude)],
                                             mode='walking'
                                            )['rows'][0]['elements'][0]
                G.add_edge(node, tgt_node, dist=dist)

                time.sleep(0.01)

    return G


def parse_digraph(edges, nodes):
    G = nx.DiGraph()
    df = pd.read_csv(nodes)
    for index, row in df.iterrows():
        G.add_node(str(row['ID']), label=row['Label'], elevation=row['Elevation'], capacity=row['Capacity'], latitude=row['Latitude'], longitude=row['Longitude'])

    df = pd.read_csv(edges)
    for index, row in df.iterrows():
        if(row['Source'] == row['Target']): continue
        start, end = row['Interval'][2:-2].split(", ")
        start = datetime.fromisoformat(start)
        end = datetime.fromisoformat(end)
        end_hour = end.hour
        if(end.hour < start.hour):
            end_hour += 24

        if(start.hour <= HOUR_TO_COLLECT <= end_hour):  # We are either started or ended in the target hour!
            G.add_edge(str(row['Source']), str(row['Target']), chosen=False)


    if(False):
        x = PrettyTable()
        x.field_names = ["ID", "Label", "Elevation", "Capacity", "In Degree"]
        for k, v in G.in_degree:
            if(v > 10):
                node = G.nodes[k]
                x.add_row([k, node['label'], node['elevation'], node['capacity'], v])
        print(x)


    return G


def plot_degree_dist(G, name):
    degrees = sorted([G.in_degree(n) for n in G.nodes()], reverse=True)
    degreeCnt = collections.Counter(degrees)
    deg, cnt = zip(*degreeCnt.items())

    plt.clf()
    plt.subplot(2,2,1)
    plt.hist(degrees, bins=10, histtype='step')
    plt.title("In Degree Histogram")

    plt.subplot(2,2,2)
    total_cnt = sum(cnt)
    pdf_cnt = [i / total_cnt for i in list(cnt)]
    plt.loglog(deg, pdf_cnt)
    plt.title("In Degree PDF")

    degrees = sorted([G.out_degree(n) for n in G.nodes()], reverse=True)
    degreeCnt = collections.Counter(degrees)
    deg, cnt = zip(*degreeCnt.items())

    plt.subplot(2,2,3)
    plt.hist(degrees)
    plt.title("Out Degree Histogram")

    plt.subplot(2,2,4)
    total_cnt = sum(cnt)
    pdf_cnt = [i / total_cnt for i in list(cnt)]
    plt.loglog(deg, pdf_cnt)
    plt.title("Out Degree PDF")

    plt.savefig(name)


def plot_congestion(G, name="", pre=True):
    congestion = sorted([round((G.in_degree(n) - G.out_degree(n)) / G.nodes[n]['capacity'], 2) * 100 for n in G.nodes()], reverse=True)
    congestionCnt = collections.Counter(congestion)
    con, cnt = zip(*congestionCnt.items())

    plt.subplot(2,1,1 if pre else 2)
    plt.hist(congestion, bins=50, density=True, color='c', edgecolor='k')
    plt.axvline(CONGESTION_CONSIDERATION, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(-CONGESTION_CONSIDERATION, color='k', linestyle='dashed', linewidth=1)
    plt.title("{}::Congestion Histogram: HOUR {}: Compliance {}%".format("BEFORE" if pre else "AFTER", HOUR_TO_COLLECT, COMPLIANCE_RATE * 100))
    plt.xlabel("% of congestion, In - Out / Capacity")
    plt.ylabel("Probability Distribution")
    
    '''
    plt.subplot(2,2,2)
    total_cnt = sum(cnt)
    pdf_cnt = [i / total_cnt for i in list(cnt)]
    plt.loglog(con, pdf_cnt)
    plt.title("Full Congestion PDF: HOUR {}".format(HOUR_TO_COLLECT))

    congestion = sorted([round((G.out_degree(n) - G.in_degree(n)) / G.nodes[n]['capacity'], 2) * 100 for n in G.nodes()], reverse=True)
    congestionCnt = collections.Counter(congestion)
    con, cnt = zip(*congestionCnt.items())

    plt.subplot(2,2,3 if pre else 4)
    plt.hist(congestion, bins=50, density=True, color='c', edgecolor='k')
    plt.axvline(CONGESTION_CONSIDERATION, color='k', linestyle='dashed', linewidth=1)
    plt.title("{}::Empty Congestion Histogram: HOUR {}".format("BEFORE" if pre else "AFTER", HOUR_TO_COLLECT))

    plt.subplot(2,2,4)
    total_cnt = sum(cnt)
    pdf_cnt = [i / total_cnt for i in list(cnt)]
    plt.loglog(con, pdf_cnt)
    plt.title("Empty Congestion PDF: HOUR {}".format(HOUR_TO_COLLECT))
    '''

    if(not(pre)):
        plt.savefig(name)

def calc_congestion_drain(G, n, addl=0):
    capacity = G.nodes[n]['capacity']
    in_degree = G.in_degree(n) + addl
    out_degree = G.out_degree(n)

    return (in_degree - out_degree / capacity) * 100


def calc_congestion_source(G, n, addl=0):
    capacity = G.nodes[n]['capacity']
    out_degree = G.out_degree(n) + addl
    in_degree = G.in_degree(n)

    return (out_degree - in_degree / capacity) * 100


def find_nn(prox_G, n):
    list_of_nn = []
    for edge in prox_G.edges(n):
        list_of_nn.append(edge[0] if edge[1] == n else edge[1])
    return list_of_nn


def optimize(G, prox_G, source, drain):
    source_list = [source, *find_nn(prox_G, source)] if type(source) is str else source
    dest_list = [drain, *find_nn(prox_G, drain)] if type(drain) is str else drain

    if(type(drain) is list):  # to avoid redundancy
        if(source_list[0] in dest_list):  # If the drain is part of source_list, remove drain from source
            dest_list.remove(source_list[0])
        source_list = [x for x in source_list if x not in dest_list]
    else:
        if(dest_list[0] in source_list):
            source_list.remove(dest_list[0])
        dest_list = [x for x in dest_list if x not in source_list]

    d_a = [0] + [WEIGHT_S_a * ((prox_G.edges[(k, source_list[0])]['dist']['distance']['value'] / SCALE_d) ** 2) for k in source_list[1:]]
    d_b = [0] + [WEIGHT_b_D * ((prox_G.edges[(k, dest_list[0])]['dist']['distance']['value'] / SCALE_d) ** 2) for k in dest_list[1:]]
    C_s = [calc_congestion_source(G, k, 1 if k != source_list[0] else 0) for k in source_list]
    C_d = [calc_congestion_drain(G, k, 1 if k != source_list[0] else 0) for k in dest_list]

    C_s = [WEIGHT_a * k for k in C_s]
    C_d = [WEIGHT_b * k for k in C_d]

    e = []
    for s in source_list:
        e.append([WEIGHT_a_b * ((((G.nodes[s]['elevation']- G.nodes[d]['elevation']) + OFFSET_e) / SCALE_e) ** 2) for d in dest_list])

    sel_s, sel_d = solve(source_list, dest_list, d_a, d_b, C_s, C_d, e)

    if(sel_s == sel_d):
        print("Already optimal path")
    else:
        G.remove_edge(source_list[0], dest_list[0])
        G.add_edge(sel_s, sel_d, chosen=True)
        print("Replacing {}->{} w/ {}->{}".format(source_list[0], dest_list[0], sel_s, sel_d))


def optimize_start(G, prox_G, n, congestion_min=CONGESTION_CONSIDERATION):
    edge_list = list(G.out_edges(n))
    source_list = [n, *find_nn(prox_G, n)]
    considered = 0
    while(True):
        if(considered == len(edge_list)): break
        C_s = calc_congestion_source(G, n)
        if(C_s < congestion_min): break  # If below congestion min, do not consider this node

        num_choices = len(edge_list)
        edge = edge_list[int(random.uniform(0, num_choices))]
        if(G.edges[edge]["chosen"]): continue  # Do not choose the same one twice
        considered += 1
        G.edges[edge]["chosen"] = True
        dest = edge[1]
        if(dest == n): continue  # do not consider self loops

        optimize(G, prox_G, source_list, dest)


def optimize_stop(G, prox_G, n, congestion_min=CONGESTION_CONSIDERATION):
    edge_list = list(G.in_edges(n))
    dest_list = [n, *find_nn(prox_G, n)]
    considered = 0
    while(True):
        if(considered == len(edge_list)): break 
        C_d = calc_congestion_drain(G, n)
        if(C_d < congestion_min): break  # If below congestion min, do not consider this node

        num_choices = len(edge_list)
        edge = edge_list[int(random.uniform(0, num_choices))]
        if(G.edges[edge]["chosen"]): continue  # Do not choose the same one twice
        considered += 1
        G.edges[edge]["chosen"] = True
        source = edge[0]
        if(source == n): continue  # do not consider self loops

        optimize(G, prox_G, source, dest_list)
        


def walk_nodes(G, prox_G):
    n = '2006'  # This currently selects one node :P
    optimize_start(G, prox_G, n)
    optimize_stop(G, prox_G, n)


def random_optimization(G, prox_G, num_consideration=50, congestion_min=CONGESTION_CONSIDERATION):
    edges = list(G.edges)
    cnt = 0
    consideration = 0
    while(True):
        edge = edges[int(random.uniform(0, len(edges)))]
        consideration += 1
        if(consideration >= 2 * len(edges)): break
        if(edge[0] == edge[1]):
            edges.remove(edge)  # Remove self loops
            continue
        if(G.edges[edge]["chosen"]):continue
        C_d = calc_congestion_drain(G, edge[1])
        C_s = calc_congestion_source(G, edge[0])

        if(C_s < congestion_min and C_d < congestion_min): continue # If below congestion min, do not consider this node
        consideration = 0

        if(random.random() <= COMPLIANCE_RATE):  # Simulating refusal to comply
            optimize(G, prox_G, edge[0], edge[1])
        edges.remove(edge)

        cnt += 1
        if(cnt >= num_consideration): break
        print(cnt)


def output_nodes_congestion(di_G, name):
    with open(name, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        rows = [['ID', 'Label', 'Latitude', 'Longitude', 'Elevation', 'Capacity', 'Congestion']]
        for node in di_G.nodes(data=True):
            rows.append([node[0], node[1]['label'], node[1]['latitude'], node[1]['longitude'], node[1]['elevation'], node[1]['capacity']])
            # Round congestion so can plot nicely
            rows[-1].append(
                 round(((di_G.in_degree(node[0]) - di_G.out_degree(node[0])) / node[1]['capacity']) * 100))
        csv_writer.writerows(rows)


def output_edges(di_G, name):
    nx.write_edgelist(di_G, name, delimiter=",", data=False)
    df = pd.read_csv(name, header=None)
    df.to_csv(name, header=["Source", "Target"], index=False)


@click.command()
@click.argument("edges", required=True)
@click.argument("nodes", required=True)
@click.option("-n", "--name", help="Choose name of the output", default="output")
def cli(edges, nodes, name):
    prox_edge_list = Path("prox.edgelist")
    if(prox_edge_list.exists()):
        prox_G = nx.read_edgelist(prox_edge_list)
    else:
        input("I gotta read all this stuff from Google API... this will take a long time... continue?")
        prox_G = find_proximity_graph(nodes)
        nx.write_edgelist(prox_G, "prox.edgelist")

    di_G = parse_digraph(edges, nodes)
    output_nodes_congestion(di_G, "out_nodes_pre.csv")
    output_edges(di_G, "out_edges_pre.csv")
    # plot_degree_dist(di_G, "pre.png")
    plot_congestion(di_G, pre=True)
    # walk_nodes(di_G, prox_G)
    random_optimization(di_G, prox_G, num_consideration=10000)
    plot_congestion(di_G, name="{}.png".format(name), pre=False)
    output_nodes_congestion(di_G, "out_nodes_post.csv")
    output_edges(di_G, "out_edges_post.csv")



if(__name__ == '__main__'):
    cli()

