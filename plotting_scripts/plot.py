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

stats_list = ['Compute cycles', 'Stall cycles']

baseline = 'comp'
cado = 'cado'

def addOptions(parser):
    parser.add_argument("-i", "--input", type=str, default='../running_scripts/',
                        help="output trace log file")
    parser.add_argument("-o", "--output", type=str, default='cado',
                        help="output file")

def get_file_list(root_directory):
    log_file_list = []
    for log_file in os.listdir(root_directory):
        if log_file.endswith(".log"):
            log_file_list.append(log_file)
    return log_file_list

def get_workload_file_list(root_directory, workload_list):
    workload_file_dict = dict()
    for workload in workload_list:
        workload_root_directory = root_directory + workload
        workload_file_dict[workload] = get_file_list(workload_root_directory)
    return workload_file_dict

def generate_path_dict(root_directory, workload_file_dict):
    path_dict = dict()
    for workload in workload_file_dict:
        directory = os.path.join(root_directory, workload)
        for log_file in workload_file_dict[workload]:
            file_name = log_file.replace('.log', '')
            file_name = file_name.split('_')
            if len(file_name) == 5:
                workload_name = str(file_name[0])
                llc_size = str(file_name[1])
                num_pe = str(file_name[2])
                num_batch = str(file_name[3])
                dataflow = str(file_name[4])
                
                if num_pe not in path_dict:
                    path_dict[num_pe] = dict()
                
                if num_batch not in path_dict[num_pe]:
                    path_dict[num_pe][num_batch] = dict()
                    
                if workload_name not in path_dict[num_pe][num_batch]:
                    path_dict[num_pe][num_batch][workload_name] = dict()
                    
                if llc_size not in path_dict[num_pe][num_batch][workload_name]:
                    path_dict[num_pe][num_batch][workload_name][llc_size] = dict()

                path_dict[num_pe][num_batch][workload_name][llc_size][dataflow] = os.path.join(directory, log_file)
        
    return path_dict
        

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
    # print(log_file_name)
    # print(overall_stat)

    text.close()

    return per_layer_stat, overall_stat
    

def plot_per_layer(per_layer_file, ylabel='Speedups'):
    ymax = 2.5
    ymin = 0
    if ymax > 0:
        print(per_layer_file)
    df = pd.read_csv(per_layer_file, index_col=[0, 1])
    group_list = []
    subgrp_list = []
    for index in df.index:
        if index[0] not in group_list:
            group_list.append(index[0])
        if index[1] not in subgrp_list:
            subgrp_list.append(index[1])
    col_list = df.columns
    
    print(col_list)
    print(group_list)
    print(subgrp_list)

    ngroups = len(group_list)
    nsubgrps = len(subgrp_list)
    x = np.arange(nsubgrps)
    nbars = len(subgrp_list)
    width = (1 - 0.4)

    matplotlib.rcParams["hatch.linewidth"] = 2

    patterns = ["", "", ""]
    color_tab = ['#87CEFA', '#DBAD3D', '#1873BA']
    edge_color_tab = ['#000000', '#000000', '#000000']

    fig, axes = plt.subplots(nrows=1, ncols=ngroups, sharey=True, figsize=[10, 4])

    hdl_pair = []
    rects = []

    for g in range(0, ngroups):
        ax = axes[g]
        for i in range(0, nbars):
            height_curr = df[col_list[0]][group_list[g]][subgrp_list[i]]   # y coo
            
            pattern = '//' if subgrp_list[i] == 'GEO-MEAN' else ''
            
            rect_base = ax.bar(i,          # x coo
                            height_curr,  # y coo
                            width,
                            label=subgrp_list[i],
                            color=color_tab,
                            edgecolor=edge_color_tab,
                            hatch=pattern
                            )
            #    hatch=patterns[i]
            rects.append(rect_base)

            if height_curr > ymax:
                x_pos = i
                ax.annotate('{:,.2f}'.format(height_curr), xy=(x_pos, ymax), xytext=(0, 1.1), textcoords="offset points",rotation=90, color='black', ha='center', va='top', fontsize=15)
                         
        
        ax.set_xticks(x)
        ax.set_xticklabels(subgrp_list, rotation=90, fontsize=13)
        
        if g == 0:
            ax.set_ylabel(ylabel, fontsize=15)
        ax.set_ylim([ymin, ymax])
        
        ax.set_xlabel(group_list[g], fontsize=13)
        ax.xaxis.set_label_position('top') 

        ax.yaxis.grid(which='major', color='black', linestyle='--', alpha=.4)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    plt.savefig(per_layer_file.replace('.csv', '.png'), dpi=400)
    plt.savefig(per_layer_file.replace('.csv', '.pdf'), format='pdf')

    

def generate_log_results(num_pe, num_batch, path_dict, llc_size_list, workload_list, output_file, baseline='comp'):
    geo_mean_dict = dict()
    
    overall_file = output_file + '_' + str(num_pe) + '_' + str(num_batch) + '.csv'
    
    out = open(overall_file, "w")
    out.write('group,,Speedups')
    # out.write(baseline)
    # out.write(',')
    # out.write(cado)
    out.write(os.linesep)
    
    for llc_size in llc_size_list:
        speedup_list = []
        for workload in workload_list:
            baseline_log_file = path_dict[workload][llc_size][baseline]   
            baseline_time_list, baseline_final_time = process_log_file(baseline_log_file)

            cado_log_file = path_dict[workload][llc_size][cado]
            cado_time_list, cado_final_time = process_log_file(cado_log_file)
            
            print(baseline_time_list)
            print(cado_time_list)
            print(num_pe, num_batch, workload, llc_size, baseline_final_time, cado_final_time)
            
            for i in range(0, len(baseline_time_list)):
                print(baseline_time_list[i][0] + baseline_time_list[i][1], cado_time_list[i][0] + cado_time_list[i][1])
            
            if (cado_final_time[0] + cado_final_time[1]) == 0:
                speedup = 1
            else:
                speedup = (baseline_final_time[0] + baseline_final_time[1]) / (cado_final_time[0] + cado_final_time[1])
                speedup_list.append(speedup)

            if int(llc_size) > 128:
                out.write(str(llc_size))
                out.write('KB LLC')
            else:
                out.write(str(pow(2, int(llc_size))))
                out.write('-Way LLC')
            out.write(',')
            out.write(workload) 
            out.write(',')
            out.write(str(speedup))
            out.write(os.linesep)
            
        if int(llc_size) > 128:
            out.write(str(llc_size))
            out.write('KB LLC')
        else:
            out.write(str(pow(2, int(llc_size))))
            out.write('-Way LLC')
        out.write(',GEO-MEAN,')
        out.write(str(gmean(speedup_list)))
        out.write(os.linesep)
        
        geo_mean_dict[llc_size] = gmean(speedup_list)
        
    out.close()
    plot_per_layer(overall_file, ylabel='Speedups')
    return geo_mean_dict, overall_file

def generate_batch_log_results(num_pe, num_batch_list, path_dict, llc_size, workload_list, output_file, baseline='comp'):
    geo_mean_dict = dict()
    
    overall_file = output_file + '_' + str(llc_size) + '_' + str(num_pe) + '.csv'
    
    out = open(overall_file, "w")
    out.write('group,,Speedups')
    out.write(os.linesep)
    
    for num_batch in num_batch_list:
        speedup_list = []
        for workload in workload_list:
            baseline_log_file = path_dict[num_batch][workload][llc_size][baseline]   
            baseline_time_list, baseline_final_time = process_log_file(baseline_log_file)

            cado_log_file = path_dict[num_batch][workload][llc_size][cado]
            cado_time_list, cado_final_time = process_log_file(cado_log_file)
            
            print(baseline_time_list)
            print(cado_time_list)
            print(num_pe, num_batch, workload, llc_size, baseline_final_time, cado_final_time)
            
            for i in range(0, len(baseline_time_list)):
                print(baseline_time_list[i][0] + baseline_time_list[i][1], cado_time_list[i][0] + cado_time_list[i][1])
            
            if (cado_final_time[0] + cado_final_time[1]) == 0:
                speedup = 1
            else:
                speedup = (baseline_final_time[0] + baseline_final_time[1]) / (cado_final_time[0] + cado_final_time[1])
                speedup_list.append(speedup)

            out.write('batch_size = ' + str(num_batch))
            out.write(',')
            out.write(workload) 
            out.write(',')
            out.write(str(speedup))
            out.write(os.linesep)
            
        out.write('batch_size = ' + str(num_batch))
        out.write(',GEO-MEAN,')
        out.write(str(gmean(speedup_list)))
        out.write(os.linesep)
        
        geo_mean_dict[num_batch] = gmean(speedup_list)
        
    out.close()
    plot_per_layer(overall_file, ylabel='Speedups')
    return geo_mean_dict, overall_file


def generate_pe_log_results(num_pe_list, num_batch, path_dict, llc_size, workload_list, output_file, baseline='comp'):
    geo_mean_dict = dict()
    
    overall_file = output_file + '_' + str(llc_size) + '_' + str(num_batch) + '.csv'
    
    out = open(overall_file, "w")
    out.write('group,,Speedups')
    out.write(os.linesep)
    
    for num_pe in num_pe_list:
        speedup_list = []
        for workload in workload_list:
            baseline_log_file = path_dict[num_pe][num_batch][workload][llc_size][baseline]   
            baseline_time_list, baseline_final_time = process_log_file(baseline_log_file)

            cado_log_file = path_dict[num_pe][num_batch][workload][llc_size][cado]
            cado_time_list, cado_final_time = process_log_file(cado_log_file)
            
            print(baseline_time_list)
            print(cado_time_list)
            print(num_pe, num_batch, workload, llc_size, baseline_final_time, cado_final_time)
            
            for i in range(0, len(baseline_time_list)):
                print(baseline_time_list[i][0] + baseline_time_list[i][1], cado_time_list[i][0] + cado_time_list[i][1])
            
            if (cado_final_time[0] + cado_final_time[1]) == 0:
                speedup = 1
            else:
                speedup = (baseline_final_time[0] + baseline_final_time[1]) / (cado_final_time[0] + cado_final_time[1])
                speedup_list.append(speedup)

            out.write(str(num_pe)+'x'+ str(num_pe) + ' PEs')
            out.write(',')
            out.write(workload) 
            out.write(',')
            out.write(str(speedup))
            out.write(os.linesep)
            
        out.write(str(num_pe)+'x'+ str(num_pe) + ' PEs')
        out.write(',GEO-MEAN,')
        out.write(str(gmean(speedup_list)))
        out.write(os.linesep)
        
        geo_mean_dict[num_pe] = gmean(speedup_list)
        
    out.close()
    plot_per_layer(overall_file, ylabel='Speedups')
    return geo_mean_dict, overall_file


def process_geo_mean(result_dict, output_file, is_pe):
    print(result_dict)

    overall_file = output_file + '.csv'
    
    out = open(overall_file, "w")
    out.write('group,,Speedups')
    out.write(os.linesep)

    new_result_dict = dict()

    for pe_batch in result_dict:
        for llc_size in result_dict[pe_batch]:

            if llc_size not in new_result_dict:
                new_result_dict[llc_size] = dict()

            new_result_dict[llc_size][pe_batch] = result_dict[pe_batch][llc_size]

    print(new_result_dict)

    for llc_size in new_result_dict:
        geo_mean_list = []
        for pe_batch in new_result_dict[llc_size]:

            out.write(str(llc_size))
            
            if is_pe:
                out.write(' * n KB LLC')
                out.write(',')
                out.write(str(pe_batch))
                out.write('X')
                out.write(str(pe_batch))
                out.write(' PEs')
            else:
                out.write('KB LLC')
                out.write(',')
                out.write('BatchSize=')
                out.write(str(pe_batch))
            out.write(',')
            out.write(str(new_result_dict[llc_size][pe_batch]))
            out.write(os.linesep)


            geo_mean_list.append(float(new_result_dict[llc_size][pe_batch]))

        out.write(str(llc_size))
        if is_pe:
            out.write(' * n KB LLC')
        else:
            out.write('KB LLC')  
        out.write(',GEO-MEAN,')
        out.write(str(gmean(geo_mean_list)))
        out.write(os.linesep)

    out.close()
    plot_per_layer(overall_file)
    return
    
def plot_overall(overall_file, llc_highlight_list, llc_size_list):
    ymin = 0
    ymax = 2.5
    df = pd.read_csv(overall_file, index_col=[0, 1])
    group_list = []
    subgrp_list = []
    for index in df.index:
        if index[0] not in group_list:
            group_list.append(index[0])
        if index[1] not in subgrp_list:
            subgrp_list.append(index[1])
    col_list = df.columns
    
    print(col_list)
    print(group_list)
    print(subgrp_list)

    ngroups = len(group_list)
    nsubgrps = len(subgrp_list)
    
    index_list = []
    for i in range(0, len(group_list)):
        if group_list[i] in llc_highlight_list:
            index_list.append(group_list[i])
    

    matplotlib.rcParams["hatch.linewidth"] = 2

    color_tab = ['#87CEFA', '#DBAD3D', '#1873BA']
    edge_color_tab = ['#000000', '#000000', '#000000']

    fig, axes = plt.subplots(nrows=1, ncols=len(llc_highlight_list), sharey=True, figsize=[7, 4])


    for g in range(0, len(llc_highlight_list)):
        x = np.arange(nsubgrps-1)
        nbars = len(subgrp_list) - 1
        width = (1 - 0.4)
        
        ax = axes[g]
        for i in range(0, nbars):
            height_curr = df[col_list[0]][index_list[g]][subgrp_list[i]]   # y coo
            
            rect_base = ax.bar(i,          # x coo
                            height_curr,  # y coo
                            width,
                            label=subgrp_list[i],
                            color=color_tab,
                            edgecolor=edge_color_tab
                            )
            
            if height_curr > ymax:
                x_pos = i
                ax.annotate('{:,.2f}'.format(height_curr), xy=(x_pos, ymax), xytext=(0, 1.1), textcoords="offset points",rotation=90, color='black', ha='center', va='top', fontsize=15)
                 

        ax.set_xticks(x)
        ax.set_xticklabels(subgrp_list[:-1], rotation=90, fontsize=13)
        
        if g == 0:
            ax.set_ylabel("Speedups", fontsize=13)
        
        ax.set_xlabel(index_list[g], fontsize=13)
        ax.xaxis.set_label_position('top') 
        ax.yaxis.grid(which='major', color='black', linestyle='--', alpha=.4) 
        ax.set_ylim([ymin, ymax])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    
    overall_output_file = overall_file.replace('.csv', '_overall.csv')
    
    plt.savefig(overall_output_file.replace('.csv', '.png'), dpi=400)
    plt.savefig(overall_output_file.replace('.csv', '.pdf'), format='pdf')
    
    
    fig, ax = plt.subplots(figsize=[3, 4])
    x = np.arange(ngroups)
    nbars = ngroups
    width = (1 - 0.4)

    for i in range(0, nbars):
        height_curr = df[col_list[0]][group_list[i]][nsubgrps-1]   # y coo
            
        rect_base = ax.bar(i,          # x coo
                        height_curr,  # y coo
                        width,
                        label=subgrp_list[i],
                        color=color_tab,
                        edgecolor=edge_color_tab,
                        hatch='//'
                        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_list, rotation=90, fontsize=13)
        
    ax.set_ylabel("Speedups", fontsize=13)
    ax.set_xlabel('GEO-MEAN of 6 workloads', fontsize=13)
    ax.xaxis.set_label_position('top') 
    ax.yaxis.grid(which='major', color='black', linestyle='--', alpha=.4)

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    
    overall_output_file = overall_file.replace('.csv', '_geo_mean.csv')
    
    plt.savefig(overall_output_file.replace('.csv', '.png'), dpi=400)
    plt.savefig(overall_output_file.replace('.csv', '.pdf'), format='pdf')

def main():
    parser = argparse.ArgumentParser()
    addOptions(parser)
    
    options = parser.parse_args()
    
    input_dir = options.input
    output_file = options.output
    
    num_pe = '1'
    num_batch_list = ['1', '2', '4', '8']
    workload_list = ['alexnet', 'resnet18', 'resnet50', 'mobilenetv2', 'dlrm', 'transformer']
    llc_size_list = ['256', '512', '1024', '2048']
    
    workload_file_dict = get_workload_file_list(input_dir, workload_list)
    path_dict = generate_path_dict(input_dir, workload_file_dict)
    overall_file = dict()
    
    print(path_dict)
    
    # Scale on multi-batch
    scale_batch_result = dict()
    for num_batch in num_batch_list:
        scale_batch_result[num_batch], overall_file[num_batch] = generate_log_results(num_pe, num_batch, path_dict[num_pe][num_batch], llc_size_list, workload_list, output_file)
    batch_output_file = output_file + '_batch'
    process_geo_mean(scale_batch_result, batch_output_file, False)
    
    # Overall Results
    llc_highlight_list = ['512KB LLC', '1024KB LLC']
    plot_overall(overall_file['1'], llc_highlight_list, llc_size_list)
    
    # Scale on multi-PE
    num_pe_list = ['1', '2', '4', '8']
    num_batch = '1'
    llc_size_list = ['256', '512', '1024']
    scale_pe_result = dict()    
    for num_pe in num_pe_list:
        scale_pe_result[num_pe], overall_file = generate_log_results(num_pe, num_batch, path_dict[num_pe][num_batch], llc_size_list, workload_list, output_file)
    pe_output_file = output_file + '_pe'
    process_geo_mean(scale_pe_result, pe_output_file, True)

    # Sensitivity on differet cache assoc
    num_pe = '1'
    num_batch = '1'
    llc_size_list = ['1', '2', '3', '4']
    assoc_output_file = output_file + '_assoc'
    scale_pe_result, overall_file = generate_log_results(num_pe, num_batch, path_dict[num_pe][num_batch], llc_size_list, workload_list, assoc_output_file)


    # Sensitivity on differet batch size -- 1024 full result
    num_pe = '1'
    num_batch_list = ['1', '2', '4', '8']
    llc_size = '1024'
    batch_output_file = output_file + '_batch'
    scale_pe_result, overall_file = generate_batch_log_results(num_pe, num_batch_list, path_dict[num_pe], llc_size, workload_list, batch_output_file)

    # Sensitivity on differet PEs -- 1024 full result
    num_pe_list = ['1', '2', '4', '8']
    num_batch = '1'
    llc_size = '1024'
    pe_output_file = output_file + '_pe'
    scale_pe_result, overall_file = generate_pe_log_results(num_pe_list, num_batch, path_dict, llc_size, workload_list, pe_output_file)


    
if __name__ == '__main__':
    main()
