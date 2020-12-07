# complex-networks-citibike

## Installation

Requires these python libraries:

```console
pip install pandas click numbas numpy
```

## How to use parse.py

This got randomly complicated, so I'll be documenting how to use parse.py here. This script is intended to pull out a
bunch of data from the citibike csv as well as the elevation csv (optional). To use this script without elevation use
as follows:

```console
python parse.py {path to citibike.csv}
```

Where citibike.csv can be obtained [here] ()

This will output two csvs: nodes.csv and edges.csv which can be put into Gephi for analysis.

## Elevation extension

You can also pass in elevation to the script. This can be achieved like below:

```console
python parse.py {path to citibike.csv} -e {path to elevation.csv}
```

Where elevation.csv can be obtained [here] (https://data.cityofnewyork.us/Transportation/Elevation-points/szwg-xci6)

Since parsing the elevation csv takes a longggg time (its huge), there is an optimization that after the first time you
run it, the script will serialize the data onto a local file. Therefore, you can run future runs without the below
option. I.e.

```console
python parse.py {path to citibike.csv}
```

Note that this will still take forever, since for most nodes, we have to calculate the distance... and well you can
imagine how long it'd take to calculate the minimum distance across that huge elevation file.

If you don't care about the elvation, then get rid of the elevation pickled file or don't create one in the first place.

## Weather extension

You can also pass in weather data to the script to tag the target with PRCP, TMAX, and TMIN.

```console
python parse.py ... -w {path to weather.csv}
```

Where weather.csv can be obtained [here] (https://www.ncdc.noaa.gov/data-access)


## How to use analyze.py

This script accepts the nodes.csv and edgesXXX.csv that is created from parse.py. The current repo contains an example
that can be used directly to test the script. The prox.edgelist is required if you don't want to call Google Maps API
(for which you'd need to setup a API Key).

You can run the script as follows:

```console
python analyze.py edgesXXX.csv nodes.csv -n {name of run}
```

Option argument of --quiet if you want to disable VERBOSE logging as well as --seed if you want to pass a hex value seed
to control the randomness.

