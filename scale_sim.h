#ifndef _scale_sim_h
#define _scale_sim_h

#include <string>
#include <iostream>

#include "simulator.h"
#include "scale_config.h"
#include "topology_utils.h"

class ScaleSim
{
public:
    ScaleSim(bool verbose, char* config, char* topology, bool input_type_gemm);
    void set_params(char* config_file, char* topology_file);
    void run_scale(char* top_path);

private:
    void run_once();
    void print_run_configs();

    Config *config;
    Topology *topology;
    Simulator *simulator;

    char* config_file;
    char* topology_file;
    bool read_gemm_inputs;

    char* top_path;

    bool verbose_flag = false;
    bool run_done_flag = false;
    bool logs_generated_flag = false;

};

ScaleSim::ScaleSim(bool verbose, char* config_file, char* topology_file, bool input_type_gemm)
{
    this->verbose_flag = verbose;

    this->read_gemm_inputs = input_type_gemm;

    config = new Config();
    topology = new Topology();
    simulator = new Simulator();

    run_done_flag = false;
    logs_generated_flag = false;

    this->set_params(config_file, topology_file);
}

void ScaleSim::set_params(char* config_file, char* topology_file)
{
    this->config_file = config_file;
    this->topology_file = topology_file;
    config->read_conf_file(this->config_file);
    topology->load_arrays(config, this->topology_file, config->is_prefetch_demand());
}

void ScaleSim::run_scale(char* top_path)
{
    this->top_path = top_path;
    simulator->set_params(config, topology, top_path, verbose_flag);
    this->run_once();
}

void ScaleSim::run_once()
{
    if (verbose_flag)
    {
        print_run_configs();
    }

    this->simulator->run();

    run_done_flag = true;
    logs_generated_flag = true;

    if (verbose_flag)
    {
        printf("************ SCALE SIM Run Complete ****************\n");
    }
}

void ScaleSim::print_run_configs()
{
    string df_string = "Output Stationary"; 
    string df = this->config->get_dataflow();

    if (df == "ws")
        df_string = "Weight Stationary";
    else if (df == "is")
        df_string = "Input Stationary";

    printf("====================================================\n");
    printf("******************* SCALE SIM **********************\n");
    printf("====================================================\n");

    auto array_dims = this->config->get_array_dims();
    printf("Array Size: \t %ld X %ld \n", array_dims.arr_h, array_dims.arr_w);

    auto mem_sizes = this->config->get_mem_sizes();

    printf("SRAM IFMAP (kB): \t%ld\n", mem_sizes.ifmap_kb);
    printf("SRAM Filter (kB): \t%ld\n", mem_sizes.filter_kb);
    printf("SRAM OFMAP (kB): \t%ld\n", mem_sizes.ofmap_kb);

    printf("Dataflow: \t%s\n", df_string.c_str());
    printf("CSV file path: \t%s\n", this->config->get_topology_path().c_str());
    printf("Number of Remote Memory Banks: \t%ld\n", this->config->get_mem_banks());

    printf("Bandwidth: \t%ld\n", this->config->get_bandwidth());
    printf("====================================================\n");
}

#endif