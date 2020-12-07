'''
:File: parse.py
:Author: Jayesh Joshi
:Email: jayeshjo1@utexas.edu

Parser script intended to parse through all of the different dataset from Citibike, New York Weather, and New York
elevation to represent as a single network. This file receives as an input these following datasets:
    * NYC Citibke Monthly trip information
    * NYC Weather information
    * NYC elevation information
    * Stations.json that includes relavant metadata of each station respectively.

Firstly, this file will parse elevation information to the geolocation provided that can be used later to find the
location nearest to each station. Similarly with weather stations. To find the nearest elevation/weather station, we
search around the proximity of each station by their geolocation using circular inference and from that find the closest
location for the information.

Finally, we gather only the important features into a single set of CSV files which contains the information at each
node and edge respectively. For our project, the nodes represent the individual stations and contain metadata such as
the geolocation, capacity, elevation, and labels of that station. This is a static value so we create a single nodes
file for the whole month of trips. Then, we create a separate edges file for each day of the month, and to each edge we
tag along the metadata related to the trip, such as the starting/ending stations as well as the metadata of the day
such as weather information and holiday/weekday/weekend.

Example output:
    * edges2020-08-01.csv
    * ... (for each day)
    * nodes.csv

These edges and nodes can be plugged directly into Gephi or other network analysis tools, as well as directly to our
next set of scripts: analyze.py.
'''

import pandas
from pandas.tseries.holiday import USFederalHolidayCalendar
import click
import re
import pickle
import os
import json
from datetime import datetime
import numpy as np
from math import cos, asin, sqrt
from numba.typed import List
from numba import njit

timestamp_patt = re.compile(r"(.+) (.+)\.\d+")
timestamp_patt2 = re.compile(r"(\d+)-(\d+)-(\d+)")
geom_point_patt = re.compile(r"([-+]?[0-9]+\.[0-9]+) ([-+]?[0-9]+\.[0-9]+)")

ROUNDING_LEVEL = 5

CLEAR_SURROUND = 0.0004  # This var is used to cut off a bunch of stuff from the elevation LUT
# Too big => take too long
# Too small => too small radius to look around

US_CAL = USFederalHolidayCalendar()
US_HOLIDAYS = US_CAL.holidays()


# https://numba.pydata.org/numba-doc/latest/user/performance-tips.html
@njit  # Use Nambas NJIT optimization to optimize the runtime here... 
def distance(lat1, lon1, lat2, lon2):
    '''
    Calculate the distance between two geolocations.

    :param lat1: Latitude of the first geolocation
    :type lat1: int
    :param lon1: Longitude of the first geolocation
    :type lon1: int
    :param lat2: Latitude of the second geolocation
    :type lat2: int
    :param lon2: Longitude of the second geolocation
    :type lon2: int
    '''
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))


# https://numba.pydata.org/numba-doc/latest/user/performance-tips.html
@njit  # Use Nambas NJIT optimization to optimize the runtime here... 
def cut_data(data, ll_tuple, surround):
    '''
    Finds all of the target locations surrounding our source station measured by a distance around the source station
    calculated by surround.

    This is used to reduce our search so we don't need to calculate the distance across ALL of the different points
    in new york city -- that would potentially take hours.

    :params data: Contains all of the different tuples of geolocations that we need to search through
    :type data: np.array
    :params ll_tuple: Source geolocation. [0] represents longitude, [1] represents latitude
    :type tuple: tuple
    :params spike: Indicates how much we need to expand the region if we don't find 
    :type spike:
    '''
    new_data = List()
    lon_min = ll_tuple[0] * (1 + surround)  # assuming lon is always negative
    lon_max = ll_tuple[0] * (1 - surround)
    lat_min = ll_tuple[1] * (1 - surround)
    lat_max = ll_tuple[1] * (1 + surround)

    for lon, lat in data:
        if(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            new_data.append((lon, lat, ))

    return new_data


def closest(data, ll_tuple, spike=4):
    '''
    Finds the nearest geolocation in the data to the current geolocation. This is used to search across all of the
    geolocations that we have available to find the nearest location that we have information of each of the different
    bike stations across NYC.

    It will initially look for locations near CLEAR_SURROUND around our current geolocation. If it fails, we look at a
    larger location. This is used to reduce the number of stations under consideration to reduce the time it takes to
    calculate the distance between a node and nearby weather/elevation stations.

    :params data: Contains all of the different tuples of geolocations that we need to search through
    :type data: np.array
    :params ll_tuple: Source geolocation. [0] represents longitude, [1] represents latitude
    :type tuple: tuple
    :params spike: Indicates how much we need to expand the region if we don't find 
    :type spike:

    :returns: Closest elevation target to our source station.
    '''
    cutted_data = cut_data(data, ll_tuple, CLEAR_SURROUND)
    if(not(cutted_data)):
        cutted_data = cut_data(data, ll_tuple, CLEAR_SURROUND * spike)

    lon = ll_tuple[0]
    lat = ll_tuple[1]

    min_val = min(cutted_data, key=lambda p: distance(lat, lon, p[1], p[0]))

    return min_val


def parse_nodes(df, ele_dict, stations_dict):
    '''
    Parse through all of the different nodes in our network to find all of the relavant features, as mentioned above.

    This will parse through all of the nodes that we need for our network based on the datasets provided. If the target
    datasets are not available, it will just put empty information for that feature. We also need to impute data in the
    case that only some of the data are missing within the dataset.

    :params df: Contains the citibike dataset
    :type df: Pandas DataFrame
    :params ele_dict: Contains all of the different geolocation and elevation information across NYC
    :type ele_dict: dict
    :params stations_dict: Contains all of the capacity information for stations in NYC
    :type stations_dict: dict
    '''
    print("Collecting Nodes")

    # Begin by dropping features that we don't care about
    df = df.drop(["tripduration", "starttime", "stoptime", "bikeid", "usertype", "birth year", "gender"], axis=1)

    # Now, we aggregate the station IDs/names to their respective nodes
    start_station = df.drop_duplicates(subset="start station id")
    start_station.drop(["end station id", "end station name", "end station latitude", "end station longitude"], axis=1, inplace=True)
    start_station = start_station.rename(columns={"start station id": "ID", "start station name": "Label", "start station latitude": "Latitude", "start station longitude": "Longitude"})

    end_station = df.drop_duplicates(subset="end station id")
    end_station.drop(["start station id", "start station name", "start station latitude", "start station longitude"], axis=1, inplace=True)
    end_station = end_station.rename(columns={"end station id": "ID", "end station name": "Label", "end station latitude": "Latitude", "end station longitude": "Longitude"})

    # This includes the set of complete stations that we employ on our network. And contain all of the features so
    # far by aggregating the information directly from citibike
    complete_station = pandas.concat([start_station, end_station], ignore_index=True).drop_duplicates(subset="ID")


    # This sectioni s to gather all of the relavant elevation information
    ele_list = []
    ele_ll = np.array(list(ele_dict.keys()))

    if(ele_dict):
        print("\nFinding the closest elevation data, this'll take a while...")
        with click.progressbar(length=len(complete_station["Latitude"])) as bar:
            for idx, row in complete_station.iterrows():
                ll_tuple = (round(float(row["Longitude"]), ROUNDING_LEVEL), round(float(row["Latitude"]), ROUNDING_LEVEL), )

                if (ll_tuple in ele_dict):
                    ele_list.append(ele_dict[ll_tuple])
                else:
                    ele_list.append(ele_dict[closest(ele_ll, ll_tuple)])

                bar.update(1)
    else:
        ele_list = [0] * len(complete_station["Longitude"])

    # Next, we need to find capacity information for each station.
    stations_list = []
    if(stations_dict):
        stations_avg = int(np.average([v for k,v in stations_dict.items()]))
        for idx, row in complete_station.iterrows():
            # if the capacity information is missing for a target station, just use the average capacity across all
            # stations
            if(row['ID'] not in stations_dict or stations_dict[row['ID']] == 0):
                stations_list.append(stations_avg)
            else:
                stations_list.append(stations_dict[row['ID']])
    else:
        stations_list = [0] * len(complete_station['Longitude'])
    
    complete_station["Elevation"] = ele_list
    complete_station["Capacity"] = stations_list

    # Save off to file
    complete_station.to_csv("nodes.csv", index=False)


def find_timestamp(timestamp):
    '''
    Convert timestamp to Gephi friendly timestamp

    :params timestamp: Timestamp read directly from the citibike dataset
    :type timestamp: str
    '''
    patt = timestamp_patt.search(timestamp)
    return "{}T{}".format(patt.group(1), patt.group(2))


def find_weekday(timestamp):
    '''
    Find whether the timestamp belongs to a weekday.

    :params timestamp: Timestamp read directly from the citibike dataset
    :type timestamp: str
    '''
    year, month, day = [int(k) for k in timestamp_patt2.search(timestamp).groups()]
    m_time = datetime(year, month, day)

    return 0 <= m_time.isoweekday() <= 5


def parse_edges_target(df, target=""):
    '''
    Apply different filters to each of the features that are relavant to our network, to convert each of the different
    features in the NYC Citibike Dataset to information we can use directly for our analysis.

    Then save off to a file

    :params df: Citibike Dataset
    :type df: DataFrame
    :params target: Name that we want to give to the current set of edges (as timestamp)
    :type target: str
    '''
    if(target):
        print("Collecting Edges for {}".format(target))
    df = df.rename(columns={"end station id": "Target", "start station id": "Source"})
    df["starttime"] = df["starttime"].apply(find_timestamp)
    df["stoptime"] = df["stoptime"].apply(find_timestamp)
    df["Interval"] = df[['starttime', 'stoptime']].apply(lambda x: ', '.join(x), axis=1)
    df["Interval"] = df["Interval"].apply(lambda x: "<[{}]>".format(x))
    df["Weekday"] = df["starttime"].apply(find_weekday)
    df["Holiday"] = df["starttime"].apply(lambda k: k in US_HOLIDAYS)
    df = df.drop(["starttime", "stoptime"], axis=1, inplace=False)
    df.to_csv("edges{}.csv".format(target), index=False)


def gather_weather(df, weather_dict={}):
    '''
    Add weather information related to the current edge by finding the nearest weather station

    :params df: NYC weather Dataset
    :type df: DataFrame
    :params weather_dict: 
    :type weather_dict: dict
    '''
    if(weather_dict):
        print("Collecting Weather Information")
        pcp, tmax, tmin = [], [], []
        weather_ll = np.array(list(weather_dict.keys()))
        quick_lu = {}

        with click.progressbar(length=len(df.index)) as bar:
            for index, row in df.iterrows():
                lat, lon = float(row["start station latitude"]), float(row["start station longitude"])
                date = row["starttime"]
                year, month, day = [int(k) for k in timestamp_patt2.search(date).groups()]

                if((lat, lon) not in quick_lu):
                    quick_lu[(lat, lon,)] = weather_dict[closest(weather_ll, (lon, lat, ), spike=20)]

                closest_station = quick_lu[(lat, lon)][(year, month, day, )]

                pcp.append(closest_station[0])            
                tmax.append(closest_station[1])
                tmin.append(closest_station[2])

                bar.update(1)

        df["PRCP"] = pcp
        df["TMAX"] = tmax
        df["TMIN"] = tmin
            

def parse_edges(df, per_day=False, weather_dict={}):
    '''
    Parse through the dataset to find all of the relavant features for each individual trips as needed by our network.

    This can parse it off into edges separate across all of the days of a month or for the entire month.

    :params df: NYC weather Dataset
    :type df: DataFrame
    :params per_day: Used to determine if we need to break up the edges per day.
    :type per_day: bool
    :params weather_dict: 
    :type weather_dict: dict
    '''
    print("Collecting Edges")
    gather_weather(df, weather_dict)
    df = df.drop(["tripduration", "bikeid", "usertype", "birth year", "gender", "start station latitude", "start station longitude", "start station name", "end station latitude", "end station longitude", "end station name"], axis=1, inplace=False)

    if(per_day):
        # if per day, we need to parse the start/stop times to find the relavant days across eht entire month
        df[['Day', 'null']] = df.stoptime.str.split(expand=True)
        df[['Day', 'null']] = df.starttime.str.split(expand=True)
        df = df.drop(['null'], axis=1, inplace=False)
        dfs = dict(tuple(df.groupby("Day")))

        for key, df in dfs.items():
            parse_edges_target(df, key)
    else:
        parse_edges_target(df)


def parse_elevation(elevation_src):
    '''
    Parse through all of the elevation information to get a a datastructure that we can work with: tuples of geolocation
    with the elevation as key.

    :params elevation_src: Path to the elevations information csv
    :type elevation_src: str
    '''
    ele_dict = {}

    print("Parsing through Elevation")
    if(elevation_src):  # Holy hell this takes forever... hopefully pickling saves some time
        df = pandas.read_csv(elevation_src)

        for idx, row in df.iterrows():
            geom = row['the_geom']
            longitude, latitude = [round(float(i), ROUNDING_LEVEL) for i in geom_point_patt.search(geom).groups()]
            ele_dict[(longitude, latitude, )] = row["ELEVATION"]
        
        pickle.dump(ele_dict, open("elevation_pickled.pickle", "wb"))
    elif(os.path.exists("elevation_pickled.pickle")):  # Heck yeah, we only ever need to do that once
        ele_dict = pickle.load(open("elevation_pickled.pickle", "rb"))

    return ele_dict


def parse_weather(weather_src):
    '''
    Parse through all of the elevation information to get a a datastructure that we can work with: tuples of geolocation
    with the weather information as prcp (percipitation), tmax/tmin (temp max/min) of that day, for each day of the
    month.

    :params weather_src: Path to the weather information csv
    :type weather_src: str
    '''
    weather_dict = {}
    
    print("Parsing through Weather")
    
    if(weather_src):
        df = pandas.read_csv(weather_src)
        df.dropna(subset=["PRCP", "TMAX", "TMIN"], inplace=True)

        for idx, row in df.iterrows():
            prcp, tmax, tmin = float(row["PRCP"]), int(row["TMAX"]), int(row["TMIN"])

            year, month, day = [int(k) for k in timestamp_patt2.search(row["DATE"]).groups()]
            lat, lon = float(row["LATITUDE"]), float(row["LONGITUDE"])
            weather_dict.setdefault((lon, lat, ), {})[(year, month, day)] = (prcp, tmax, tmin,)

    return weather_dict


def parse_stations(stations_src):
    '''
    Parse through the stations information to gather the capacity/etc information that can be easily used as LUT for
    our aggregation

    :params stations_src: Path to the stations information csv
    :type stations_src: str
    '''
    station_dict = {}

    print("Parsing through Stations")

    if(stations_src):
        m_json = json.load(open(stations_src, "r"))
        for feature in m_json["features"]:
            station_id = feature['properties']['station_id']
            capacity = feature['properties']['capacity']
            station_dict[int(station_id)] = int(capacity)
    
    return station_dict


@click.command()
@click.argument('src', required=True)
@click.option("-e", "--elevation", help="Point to the elevation CSV")
@click.option("-w", "--weather", help="Point to the weather CSV")
@click.option("-s", "--stations", help="Point to stations.json")
@click.option("--per_day", help="Choose to break the data into per-day basis", is_flag=True, default=False)
@click.option("--skip_nodes", help="Choose to skip nodes generation", is_flag=True, default=False)
@click.option("--skip_edges", help="Choose to skip edges generation", is_flag=True, default=False)
def cli(src, elevation, weather, stations, per_day, skip_nodes, skip_edges):
    '''
    Click command line interface to retrieve targets/sources from command line
    '''
    ele_dict = parse_elevation(elevation)
    weather_dict = parse_weather(weather)
    stations_dict = parse_stations(stations)

    print("Parsing {}".format(src))
    df = pandas.read_csv(src)

    if(not(skip_nodes)):
        parse_nodes(df, ele_dict, stations_dict)
    if(not(skip_edges)):
        parse_edges(df, per_day, weather_dict)


if(__name__ == '__main__'):
    cli()

