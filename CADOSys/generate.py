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

from subprocess import Popen, PIPE

from scipy.stats import gmean

import configparser

info_keys = ['Layer Type', 'IFMAP Height', 'IFMAP Width', 'Filter Height', 'Filter Width', 'Channels', 'Num Filter', 'Stride Height', 'Stride Width', 'IFMAP Source', 'Filter Source', 'PE']
shape_keys = ['ifmap_op_mat_H', 'ifmap_op_mat_W', 'filter_op_mat_H', 'filter_op_mat_W', 'ofmap_op_mat_H', 'ofmap_op_mat_W']


def addOptions(parser):
    parser.add_argument("-i", "--input", type=str, default='../topologies/conv_nets/test_resnet18.csv',
                        help="intput worload csv file")
    parser.add_argument("-c", "--cfg", type=str, default='../configs/scale.cfg',
                        help="intput cfg file")
    parser.add_argument("-s", "--shape", type=str, default='../run_scripts/resnet18/resnet18_c256_is_shape.csv',
                        help="intput shape csv file")
    parser.add_argument("-o", "--outputCADO", type=str, default='../topologies/conv_nets/test_resnet18_cado.csv',
                        help="output file")
    parser.add_argument("-p", "--outputCOMP", type=str, default='../topologies/conv_nets/test_resnet18_comp.csv',
                        help="output file")


def get_sys_info(cfg_file, num_pe):
    config = configparser.ConfigParser()
    # config.optionxform = str
    config.read(cfg_file)

    print(config.sections())

    arr_h = int(config['architecture_presets']['ArrayHeight'])
    arr_w = int(config['architecture_presets']['ArrayWidth'])
    word_size = int(config['architecture_presets']['WordSize'])

    cache_capacity = int(config['llc']['SizekB']) * 1024
    cache_assoc = int(pow(2, int(config['llc']['Assoc'])))
    batch_size = int(config['architecture_presets']['BatchSize'])

    return arr_h, arr_w, cache_capacity, cache_assoc, word_size, batch_size

def convert_csv_to_dict(csvFilePath):
    data = {}

    with open(csvFilePath) as csvf:
        csvReader = csv.DictReader(csvf)
         
        for rows in csvReader:
            key = rows['Layer name']
            data[key] = rows

    return data

def generateCOMP(arr_h, arr_w, batch_size, shape, num_pe, num_batch, outputCOMP_dict, outputCOMP):
    for layer in outputCOMP_dict:
        if layer not in shape:
            outputCOMP_dict[layer]['Dataflow'] = 'os'
        else:
            W_M = int(shape[layer]['ifmap_op_mat_H']) * batch_size
            W_K = int(shape[layer]['ifmap_op_mat_W'])
            W_N = int(shape[layer]['filter_op_mat_W'])
            
            IS_Sr = W_K
            IS_Sc = W_M
            IS_T = W_N
            IS_row_fold = int ((IS_Sr + arr_h - 1) / arr_h)
            IS_col_fold = int ((IS_Sc + arr_w - 1) / arr_w)
            IS_T_fold = int ((IS_T + arr_h - 1))
            IS_ops = IS_row_fold * IS_col_fold * IS_T_fold
            
            WS_Sr = W_K
            WS_Sc = W_N
            WS_T = W_M
            WS_row_fold = int ((WS_Sr + arr_h - 1) / arr_h)
            WS_col_fold = int ((WS_Sc + arr_w - 1) / arr_w)
            WS_T_fold = int ((WS_T + arr_h - 1))
            WS_ops = WS_row_fold * WS_col_fold * WS_T_fold
            
            OS_Sr = W_M
            OS_Sc = W_N
            OS_T = W_K
            OS_row_fold = int ((OS_Sr + arr_h - 1) / arr_h)
            OS_col_fold = int ((OS_Sc + arr_w - 1) / arr_w)
            OS_T_fold = int ((OS_T + arr_h - 1))
            OS_ops = OS_row_fold * OS_col_fold * OS_T_fold  
            
            if WS_ops <= IS_ops and WS_ops <= OS_ops:
                outputCOMP_dict[layer]['Dataflow'] = 'ws'
            elif OS_ops <= IS_ops and OS_ops <= WS_ops:
                outputCOMP_dict[layer]['Dataflow'] = 'os'
            else:
                outputCOMP_dict[layer]['Dataflow'] = 'is'

            if (WS_ops <= IS_ops and WS_ops >= OS_ops) or (WS_ops <= OS_ops and WS_ops >= IS_ops):
                outputCOMP_dict[layer]['Second'] = 'ws'
            elif (OS_ops <= IS_ops and OS_ops >= WS_ops) or (OS_ops <= WS_ops and OS_ops >= IS_ops):
                outputCOMP_dict[layer]['Second'] = 'os'
            else:
                outputCOMP_dict[layer]['Second'] = 'is'
                
            print(IS_ops, WS_ops, OS_ops, outputCOMP_dict[layer]['Dataflow'])

    # print(outputCOMP_dict)
    
    f = open(outputCOMP, "w")
    index_str = 'Layer name'
    for ele in info_keys:
        index_str += ','
        index_str += ele
    index_str += ',Dataflow'
    f.write(index_str)
    f.write('\n')

    for layer in outputCOMP_dict:
        content_str = ''
        for ele in outputCOMP_dict[layer]:
            if ele == 'PE':
                content_str += str(num_pe)
            else:
                if ele != 'Second':
                    content_str += outputCOMP_dict[layer][ele]
            if ele != 'Dataflow':
                if ele != 'Second':
                    content_str += ','
        f.write(content_str)
        f.write('\n')
    f.close()

    return outputCOMP_dict

    

def generateCADO(outputCOMP_dict, arr_h, arr_w, cache_capacity, cache_assoc, word_size, batch_size, shape, num_pe, num_batch, outputCADO_dict, outputCADO):
    cache_capacity = cache_capacity * (cache_assoc - 1) / cache_assoc
    cache_capacity = cache_capacity
    cur_cache = cache_capacity / word_size
    for layer in outputCADO_dict:
        
        if layer not in shape:
            outputCADO_dict[layer]['Dataflow'] = 'os'
        else:
            W_M = int(shape[layer]['ifmap_op_mat_H']) * batch_size
            W_K = int(shape[layer]['ifmap_op_mat_W'])
            W_N = int(shape[layer]['filter_op_mat_W'])
            
            comp_dataflow = outputCOMP_dict[layer]['Dataflow']
            comp_second_dataflow = outputCOMP_dict[layer]['Second']

            if comp_dataflow == 'ws' and (W_N + arr_h) * W_M <= cur_cache:
                outputCADO_dict[layer]['Dataflow'] = 'ws'
                print('consistent ws')
            elif comp_dataflow == 'is' and (W_K + arr_h) * W_N <= cur_cache:
                outputCADO_dict[layer]['Dataflow'] = 'is'
                print('consistent is')
            elif comp_dataflow == 'os' and (W_N + arr_h) * W_K <= cur_cache:
                outputCADO_dict[layer]['Dataflow'] = 'os'
                print('consistent os')
            else:
                # if comp_second_dataflow == 'ws' and (W_N + arr_h) * W_M <= cur_cache: # col major
                #     outputCADO_dict[layer]['Dataflow'] = 'ws'
                # elif comp_second_dataflow == 'is' and (W_K + arr_h) * W_N <= cur_cache: # row major
                #     outputCADO_dict[layer]['Dataflow'] = 'is'
                # elif comp_second_dataflow == 'os' and (W_N + arr_h) * W_K <= cur_cache: # row major
                #     outputCADO_dict[layer]['Dataflow'] = 'os'
                if (W_N + arr_h) * W_M <= cur_cache: # col major
                    outputCADO_dict[layer]['Dataflow'] = 'ws'
                elif (W_K + arr_h) * W_N <= cur_cache: # row major
                    outputCADO_dict[layer]['Dataflow'] = 'is'
                elif (W_N + arr_h) * W_K <= cur_cache: # row major
                    outputCADO_dict[layer]['Dataflow'] = 'os'
                else:
                    if comp_dataflow == 'ws' and (2 * W_M * arr_h + arr_h * arr_w) <= cur_cache:
                        outputCADO_dict[layer]['Dataflow'] = 'ws'
                        print('consistent ws')
                    elif comp_dataflow == 'is' and (2 * W_N * arr_h + arr_h * arr_w) <= cur_cache:
                        outputCADO_dict[layer]['Dataflow'] = 'is'
                        print('consistent is')
                    elif comp_dataflow == 'os' and (2 * W_K * arr_h + arr_h * arr_w) <= cur_cache:
                        outputCADO_dict[layer]['Dataflow'] = 'os'
                        print('consistent os')
                    else:
                        if comp_second_dataflow == 'ws' and (2 * W_M * arr_h + arr_h * arr_w) <= cur_cache:
                            outputCADO_dict[layer]['Dataflow'] = 'ws'
                        elif comp_second_dataflow == 'is' and (2 * W_N * arr_h + arr_h * arr_w) <= cur_cache:
                            outputCADO_dict[layer]['Dataflow'] = 'is'
                        elif comp_second_dataflow == 'os' and (2 * W_K * arr_h + arr_h * arr_w) <= cur_cache:
                            outputCADO_dict[layer]['Dataflow'] = 'os'
                        else:
                            print('final')
                            outputCADO_dict[layer]['Dataflow'] = outputCADO_dict[layer]['Dataflow']

    f = open(outputCADO, "w")
    index_str = 'Layer name'
    for ele in info_keys:
        index_str += ','
        index_str += ele
    index_str += ',Dataflow'
    f.write(index_str)
    f.write('\n')

    for layer in outputCADO_dict:
        content_str = ''
        for ele in outputCADO_dict[layer]:
            if ele == 'PE':
                content_str += str(num_pe)
            else:
                if ele != 'Second':
                    content_str += outputCADO_dict[layer][ele]
            if ele != 'Dataflow':
                if ele != 'Second':
                    content_str += ','
        f.write(content_str)
        f.write('\n')
    f.close()


def process(input_csv, input_cfg, input_shape_file, num_pe, num_batch, outputCADO, outputCOMP):

    arr_h, arr_w, cache_capacity, cache_assoc, word_size, batch_size = get_sys_info(input_cfg, num_pe)

    input = convert_csv_to_dict(input_csv)
    shape = convert_csv_to_dict(input_shape_file)

    outputCADO_dict = input
    outputCOMP_dict = input

    # print(input)
    # print(shape)

    outputCOMP_dict = generateCOMP(arr_h, arr_w, batch_size, shape, num_pe, num_batch, outputCOMP_dict, outputCOMP)
    generateCADO(outputCOMP_dict, arr_h, arr_w, cache_capacity, cache_assoc, word_size, batch_size, shape, num_pe, num_batch, outputCADO_dict, outputCADO)


def main():
    parser = argparse.ArgumentParser()
    addOptions(parser)
    
    options = parser.parse_args()

    input_csv = options.input
    input_cfg = options.cfg
    input_shape_file = options.shape

    outputCADO = options.outputCADO
    outputCOMP = options.outputCOMP
    
    # parser.add_argument("-i", "--input", type=str, default='../topologies/conv_nets/test_resnet18.csv',
    #                     help="intput worload csv file")
    # parser.add_argument("-c", "--cfg", type=str, default='../configs/scale.cfg',
    #                     help="intput cfg file")
    # parser.add_argument("-s", "--shape", type=str, default='../run_scripts/resnet18/resnet18_c256_is_shape.csv',
    #                     help="intput shape csv file")
    # parser.add_argument("-o", "--outputCADO", type=str, default='../topologies/conv_nets/test_resnet18_cado.csv',
    #                     help="output file")
    # parser.add_argument("-p", "--outputCOMP", type=str, default='../topologies/conv_nets/test_resnet18_comp.csv',
    #                     help="output file")
    
    workload_list = ['alexnet', 'resnet18', 'resnet50', 'mobilenetv2', 'dlrm', 'transformer']
    # workload_list = ['alexnet', 'resnet18', 'mobilenetv2', 'dlrm', 'transformer']
    capacity_list = [256, 512, 1024, 2048]
    
    num_pe_list = [1]
    num_batch_list = [1, 2, 4, 8, 16, 32]

    for num_pe in num_pe_list:
        for num_batch in num_batch_list:    
            for workload in workload_list:
                for capacity in capacity_list:
                    input_csv = '../topologies/conv_nets/test_' + workload + '.csv'
                    input_cfg = '../configs/' + workload + '/' + workload + '_c' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_ws.cfg'
                    input_shape_file = '../run_scripts/' + workload + '/' + workload + '_c256_ws_shape.csv'

                    outputCADO = '../topologies/cado/' + workload + '_' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_cado.csv'
                    outputCOMP = '../topologies/cado/' + workload + '_' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_comp.csv'
                    
                    process(input_csv, input_cfg, input_shape_file, num_pe, num_batch, outputCADO, outputCOMP)
     
                    
    num_pe_list = [1, 2, 4, 8]
    num_batch_list = [1]

    for num_pe in num_pe_list:
        for num_batch in num_batch_list:    
            for workload in workload_list:
                for capacity in capacity_list:
                    input_csv = '../topologies/conv_nets/test_' + workload + '.csv'
                    input_cfg = '../configs/' + workload + '/' + workload + '_c' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_ws.cfg'
                    input_shape_file = '../run_scripts/' + workload + '/' + workload + '_c256_ws_shape.csv'

                    outputCADO = '../topologies/cado/' + workload + '_' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_cado.csv'
                    outputCOMP = '../topologies/cado/' + workload + '_' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_comp.csv'
                    
                    process(input_csv, input_cfg, input_shape_file, num_pe, num_batch, outputCADO, outputCOMP)
                    
    num_pe_list = [1]
    num_batch_list = [1]
    capacity_list = [1, 2, 3, 4]

    for num_pe in num_pe_list:
        for num_batch in num_batch_list:    
            for workload in workload_list:
                for capacity in capacity_list:
                    input_csv = '../topologies/conv_nets/test_' + workload + '.csv'
                    input_cfg = '../configs/' + workload + '/' + workload + '_c' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_ws.cfg'
                    input_shape_file = '../run_scripts/' + workload + '/' + workload + '_c256_ws_shape.csv'

                    outputCADO = '../topologies/cado/' + workload + '_' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_cado.csv'
                    outputCOMP = '../topologies/cado/' + workload + '_' + str(capacity) + '_' + str(num_pe) + '_' + str(num_batch) + '_comp.csv'
                    
                    process(input_csv, input_cfg, input_shape_file, num_pe, num_batch, outputCADO, outputCOMP)

if __name__ == '__main__':
    main()
