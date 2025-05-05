import os
import argparse
import sys
import subprocess
import psutil

import os
import collections
import csv

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import xlsxwriter
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as mtick

from subprocess import Popen, PIPE

from scipy.stats import gmean

def run_scale(csv_file):
    root_path = os.getenv("CADOSys_ROOT")
    run_cmd = root_path
    run_cmd += '/scale '
    run_cmd += csv_file
    run_cmd += ' '
    run_cmd += root_path
    run_cmd += '/configs/alexnet/alexnet_c512_1_1_ws.cfg > '
    log_file = csv_file.replace('/config/', '/output/')
    log_file = log_file.replace('.csv', '.log')
    run_cmd += log_file
    os.system(run_cmd)


def main():
    csv_file_list = []
    root_directory = './config/'
    for csv_file in os.listdir(root_directory):
        if csv_file.endswith(".csv"):
            csv_file_list.append(os.path.join(root_directory, csv_file))
    for csv_file in csv_file_list:
        print(csv_file)
        run_scale(csv_file)

if __name__ == '__main__':
    main()