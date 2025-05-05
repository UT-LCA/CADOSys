import os
import collections
import csv

import numpy as np
import xlsxwriter

import argparse
import sys
import subprocess
import psutil

from subprocess import Popen, PIPE

tmp_file = 'tmp.sh'

def addOptions(parser):
    parser.add_argument("-c", "--copy", type=str, default='resnet18',
                        help="input folder")


def IsNumericBase(s, base):
    try:
        v = int(s, base)
        return v
    except ValueError:
        return -1


def generate_workload_dict(root_directory):
    workload_dict = dict()
    workload_list = []
    for file in os.listdir(root_directory):
        d = os.path.join(root_directory, file)
        if os.path.isdir(d) and '__pycache__' not in d:
            workload_dict[file] = []
            workload_list.append(file)
    return workload_dict, workload_list


def generate_sh_file_list(root_directory):
    workload_info_list = []

    for root, directories, files in os.walk(root_directory, topdown=False):
        for name in files:
            if name.endswith('.sh'):
                workload_info_list.append(os.path.join(root_directory, name))
    return workload_info_list


def generate_all_workload_file_dict(root_directory):
    workload_dict, workload_list = generate_workload_dict(root_directory)
    
    for workload in workload_dict:
        workload_info_list = generate_sh_file_list(workload)
        workload_dict[workload] = workload_info_list

    return workload_dict, workload_list


def replace_file(copy_from, file, workload):
    
    input = open(file, "r")
    output = open(tmp_file, "w")

    for line in input:
        tmp_line = line
        tmp_line = tmp_line.replace(copy_from, workload)
        output.write(tmp_line)
        
    input.close()
    output.close()


def process_copy_replace(copy_from, workload_list, all_workload_file_dict):
    if copy_from not in workload_list:
        return
    
    workload_list.remove(copy_from)
    
    for file in all_workload_file_dict[copy_from]:
        for workload in workload_list:
            replace_file(copy_from, file, workload)
            copy_from_file_name = file.split('/')[-1]
            des_file = file.replace(copy_from, workload)
            
            cp_file_cmd = 'cp '
            cp_file_cmd += tmp_file
            cp_file_cmd += ' '
            cp_file_cmd += des_file
            print(cp_file_cmd)
            os.system(cp_file_cmd)
            
            chmod_cmd = 'chmod a+x '
            chmod_cmd += des_file
            print(chmod_cmd)
            os.system(chmod_cmd)
            
    os.remove(tmp_file)


def main():
    parser = argparse.ArgumentParser(description="Copy from")
    addOptions(parser)

    options = parser.parse_args()
    copy_from = options.copy
    
    all_workload_file_dict, workload_list = generate_all_workload_file_dict('./')
    
    print(all_workload_file_dict, workload_list)
    
    process_copy_replace(copy_from, workload_list, all_workload_file_dict)

    # all_workload_result_dict = generate_all_workload_result_dict(
    #     all_workload_info_dict)
    # print(all_workload_result_dict)



if __name__ == '__main__':
    main()
