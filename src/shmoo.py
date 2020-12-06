import click
import analyze
import params
import csv
import numpy as np
from prettytable import PrettyTable


@click.command()
@click.argument("edges", required=True)
@click.argument("nodes", required=True)
@click.option("-n", "--name", default="Temp")
@click.option("--verbose/--quiet", help="Choose whether to output verbose mode or not", default=True, is_flag=True)
def cli(edges, nodes, name, verbose):
    params.VERBOSE = verbose
    CP_list = [15]
    HOUR_LIST = [i + 8 for i in range(12)]
    compliance_rates = list(range(10, 100 + 1, 10))

    x = PrettyTable()
    x.field_names = ["Hour", "CP", "Compliance Rate", "Congestion State"]

    for idx, hour in enumerate(HOUR_LIST):
        params.HOUR_TO_COLLECT = hour
        results = analyze.run_multiple(edges, nodes, CP_list, compliance_rates, iterations=20)

        x.add_row([hour, "Default", "0%", results["def"]])
        for cp, cp_dict in results['results'].items():
            for cr, values in cp_dict.items():
                x.add_row([hour, f"{cp}%", f"{cr}%", np.average(values)])
    print(x)
    result = [tuple(filter(None, map(str.strip, splitline))) for line in str(x).splitlines() for splitline in [line.split("|")] if len(splitline) > 1]

    with open(f'{name}.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)


if(__name__ == '__main__'):
    cli()

