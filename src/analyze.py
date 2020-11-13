import click
import networkx as nx
import pandas as pd
import numpy as np
import googlemaps
import time
import collections
from datetime import datetime
from pathlib import Path
from numba.typed import List
from numba import njit
from math import cos, asin, sqrt
from prettytable import PrettyTable
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12, 12


# These are for the finding proximity
CLEAR_SURROUND = 0.0001  # This var is used to cut off a bunch of stuff from the elevation LUT
MIN_SURROUND = 5
MAX_SURROUND = 10

# These are etc
HOUR_TO_COLLECT = 11

START_CONGESTION = 5


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
        G.add_node(row['ID'], label=row['Label'], elevation=row['Elevation'], capacity=row['Capacity'])

    df = pd.read_csv(edges)
    for index, row in df.iterrows():
        start, end = row['Interval'][2:-2].split(", ")
        start = datetime.fromisoformat(start)
        end = datetime.fromisoformat(end)
        end_hour = end.hour
        if(end.hour < start.hour):
            end_hour += 24

        if(start.hour <= HOUR_TO_COLLECT <= end_hour):  # We are either started or ended in the target hour!
            G.add_edge(row['Source'], row['Target'],)


    if(False):
        x = PrettyTable()
        x.field_names = ["ID", "Label", "Elevation", "Capacity", "In Degree"]
        for k, v in G.in_degree:
            if(v > 10):
                node = G.nodes[k]
                x.add_row([k, node['label'], node['elevation'], node['capacity'], v])
        print(x)


    return G


def plot_in_degree_dist(G):
    degrees = sorted([G.in_degree(n) for n in G.nodes()], reverse=True)
    degreeCnt = collections.Counter(degrees)
    deg, cnt = zip(*degreeCnt.items())

    plt.subplot(2,2,1)
    plt.hist(degrees)
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

    plt.show()


@click.command()
@click.argument("edges", required=True)
@click.argument("nodes", required=True)
def cli(edges, nodes):
    prox_edge_list = Path("prox.edgelist")
    if(prox_edge_list.exists()):
        prox_G = nx.read_edgelist(prox_edge_list)
    else:
        input("I gotta read all this stuff from Google API... this will take a long time... continue?")
        prox_G = find_proximity_graph(nodes)
        nx.write_edgelist(prox_G, "prox.edgelist")

    di_G = parse_digraph(edges, nodes)
    plot_in_degree_dist(di_G)


if(__name__ == '__main__'):
    cli()

