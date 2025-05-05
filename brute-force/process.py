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

def process_log_file(log_file_name):
    text = open(log_file_name, "r")

    per_layer_stat = []

    last_compute_cycle = 0
    last_stall_cycle = 0

    current_compute_cycle = 0
    current_stall_cycle = 0

    for line in text:
        if 'Compute cycles' in line:
            words = line.split()
            cycles = int(words[-1])
            current_compute_cycle = cycles - last_compute_cycle

            last_compute_cycle = cycles


        elif 'Stall cycles' in line:
            words = line.split()
            cycles = int(words[-1])
            current_stall_cycle = cycles - last_stall_cycle

            last_stall_cycle = cycles

            per_layer_stat.append([current_compute_cycle - current_stall_cycle, current_stall_cycle])

    overall_stat = [last_compute_cycle - last_stall_cycle, last_stall_cycle]
    text.close()

    return per_layer_stat, overall_stat


def find_min_index(lst):
    min_value = min(lst)
    min_index = lst.index(min_value)
    return min_index

def main():
    log_file_list = []
    root_directory = './output/'
    for log_file in os.listdir(root_directory):
        if log_file.endswith(".log") and '1024' in log_file:
            log_file_list.append(os.path.join(root_directory, log_file))

    time_list = []
    for log_file in log_file_list:
        # print(log_file)
        per_layer_stat, overall_stat = process_log_file(log_file)
        time_list.append(overall_stat[0] + overall_stat[1])

    print(find_min_index(time_list))
    print(log_file_list[find_min_index(time_list)])

if __name__ == '__main__':
    main()