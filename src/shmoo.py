'''
:File: shmoo.py
:Author: Jayesh Joshi
:Email: jayeshjo1@utexas.edu

Script that can shmoo different parameters of our optimization model.
'''

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
    '''
    Simply walk through all different paramters and find the congestion state across the diferent parameters.

    This is a live edit version, so the current status shows how we shmooed across different weights
    '''
    params.VERBOSE = verbose
    CP_list = [15]
    # HOUR_LIST = [i + 8 for i in range(12)]
    HOUR_LIST = [8, 18]
    WEIGHT_D_LIST = [(1/18.75), (1/37.5), (1/75), (1/150), (1/300), (1/600), (1/1200)]
    WEIGHT_E_LIST = [(1/187.5), (1/375), (1/750), (1/1500), (1/3000), (1/6000), (1/12000)]
    # compliance_rates = list(range(10, 100 + 1, 10))
    compliance_rates = [20]

    x = PrettyTable()
    x.field_names = ["Weight_D", "Weight_e", "Hour", "CP", "Compliance Rate", "Congestion State"]

    for idx, weight_d in enumerate(WEIGHT_D_LIST):
        weight_e = WEIGHT_E_LIST[idx]
        params.WEIGHT_S_a = weight_d
        params.WEIGHT_b_D = weight_d
        params.WEIGHT_a_b = weight_e
        for idx, hour in enumerate(HOUR_LIST):
            params.HOUR_TO_COLLECT = hour
            results = analyze.run_multiple(edges, nodes, CP_list, compliance_rates, iterations=20)

            x.add_row([weight_d, weight_e, hour, "Default", "0%", results["def"]])
            for cp, cp_dict in results['results'].items():
                for cr, values in cp_dict.items():
                    x.add_row([weight_d, weight_e, hour, f"{cp}%", f"{cr}%", np.average(values)])
    print(x)
    result = [tuple(filter(None, map(str.strip, splitline))) for line in str(x).splitlines() for splitline in [line.split("|")] if len(splitline) > 1]

    with open(f'{name}.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)


if(__name__ == '__main__'):
    cli()

