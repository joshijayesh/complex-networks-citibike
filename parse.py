import pandas
import click


def parse_nodes(df):
    df = df.drop(["tripduration", "starttime", "stoptime", "bikeid", "usertype", "birth year", "gender"], axis=1)
    start_station = df.drop_duplicates(subset="start station id")
    start_station.drop(["end station id", "end station name", "end station latitude", "end station longitude"], axis=1, inplace=True)
    start_station = start_station.rename(columns={"start station id": "ID", "start station name": "Label", "start station latitude": "Latitude", "start station longitude": "Longitude"})

    end_station = df.drop_duplicates(subset="end station id")
    end_station.drop(["start station id", "start station name", "start station latitude", "start station longitude"], axis=1, inplace=True)
    end_station = end_station.rename(columns={"end station id": "ID", "end station name": "Label", "end station latitude": "Latitude", "end station longitude": "Longitude"})

    complete_station = pandas.concat([start_station, end_station], ignore_index=True).drop_duplicates(subset="ID")

    complete_station.to_csv("nodes.csv", index=False)


def parse_edges(df):
    df = df.drop(["tripduration", "bikeid", "usertype", "birth year", "gender", "start station latitude", "start station longitude", "start station name", "end station latitude", "end station longitude", "end station name"], axis=1, inplace=False)

    df = df.rename(columns={"end station id": "Target", "start station id": "Source"})
    df.to_csv("edges.csv", index=False)


@click.command()
@click.argument('src', required=True)
def cli(src):
    df = pandas.read_csv(src)

    parse_nodes(df)
    parse_edges(df)


if(__name__ == '__main__'):
    cli()

