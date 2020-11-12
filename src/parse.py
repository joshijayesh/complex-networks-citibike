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


def closest(data, ll_tuple, spike=4):
    cutted_data = cut_data(data, ll_tuple, CLEAR_SURROUND)
    if(not(cutted_data)):
        cutted_data = cut_data(data, ll_tuple, CLEAR_SURROUND * spike)

    lon = ll_tuple[0]
    lat = ll_tuple[1]

    min_val = min(cutted_data, key=lambda p: distance(lat, lon, p[1], p[0]))

    return min_val


def parse_nodes(df, ele_dict, stations_dict):
    print("Collecting Nodes")
    df = df.drop(["tripduration", "starttime", "stoptime", "bikeid", "usertype", "birth year", "gender"], axis=1)
    start_station = df.drop_duplicates(subset="start station id")
    start_station.drop(["end station id", "end station name", "end station latitude", "end station longitude"], axis=1, inplace=True)
    start_station = start_station.rename(columns={"start station id": "ID", "start station name": "Label", "start station latitude": "Latitude", "start station longitude": "Longitude"})

    end_station = df.drop_duplicates(subset="end station id")
    end_station.drop(["start station id", "start station name", "start station latitude", "start station longitude"], axis=1, inplace=True)
    end_station = end_station.rename(columns={"end station id": "ID", "end station name": "Label", "end station latitude": "Latitude", "end station longitude": "Longitude"})

    complete_station = pandas.concat([start_station, end_station], ignore_index=True).drop_duplicates(subset="ID")

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

    stations_list = []
    if(stations_dict):
        stations_avg = int(np.average([v for k,v in stations_dict.items()]))
        for idx, row in complete_station.iterrows():
            if(row['ID'] not in stations_dict or stations_dict[row['ID']] == 0):
                stations_list.append(stations_avg)
            else:
                stations_list.append(stations_dict[row['ID']])
    else:
        stations_list = [0] * len(complete_station['Longitude'])
    
    complete_station["Elevation"] = ele_list
    complete_station["Capacity"] = stations_list

    complete_station.to_csv("nodes.csv", index=False)


def find_timestamp(timestamp):
    patt = timestamp_patt.search(timestamp)
    return "{}T{}".format(patt.group(1), patt.group(2))


def find_weekday(timestamp):
    year, month, day = [int(k) for k in timestamp_patt2.search(timestamp).groups()]
    m_time = datetime(year, month, day)

    return 0 <= m_time.isoweekday() <= 5


def parse_edges_target(df, target=""):
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
    print("Collecting Edges")
    gather_weather(df, weather_dict)
    df = df.drop(["tripduration", "bikeid", "usertype", "birth year", "gender", "start station latitude", "start station longitude", "start station name", "end station latitude", "end station longitude", "end station name"], axis=1, inplace=False)

    if(per_day):
        df[['Day', 'null']] = df.stoptime.str.split(expand=True)
        df[['Day', 'null']] = df.starttime.str.split(expand=True)
        df = df.drop(['null'], axis=1, inplace=False)
        dfs = dict(tuple(df.groupby("Day")))

        for key, df in dfs.items():
            parse_edges_target(df, key)
    else:
        parse_edges_target(df)


def parse_elevation(elevation_src):
    ele_dict = {}

    print("Parsing through Elevation")
    if(elevation_src):  # Holy hell this takes forever... hopefully pickling saves some time
        df = pandas.read_csv(elevation_src)

        for idx, row in df.iterrows():
            geom = row['the_geom']
            longitude, latitude = [round(float(i), ROUNDING_LEVEL) for i in geom_point_patt.search(geom).groups()]
            ele_dict[(longitude, latitude, )] = row["ELEVATION"]
        
        pickle.dump(ele_dict, open("elevation_pickled.pickle", "wb"))
    elif(os.path.exists("elevation_pickled.pickle")):
        ele_dict = pickle.load(open("elevation_pickled.pickle", "rb"))

    return ele_dict


def parse_weather(weather_src):
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

