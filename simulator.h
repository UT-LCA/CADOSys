#ifndef _simulator_h
#define _simulator_h

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "scale_config.h"
#include "topology_utils.h"
#include "layer_sim.h"

#include "memory/double_buffer_scratchpad_mem.h"

using namespace std;

class Simulator
{
public:
    Simulator();
    void set_params(Config *config, Topology *topology, char* top_path, bool verbose_flag);
    void run();

private:
    void generate_reports();
    void get_total_cycles();

    Config *config;
    Topology *topology;
    vector<DoubleBuffer*> memory_system;

    ofstream ofs;
    
    char* top_path;
    bool verbose;

    int64_t num_layers;

    vector<LayerSim *> single_layer_sim_object_list;

    bool params_set_flag;
    bool all_layer_run_done;
};

Simulator::Simulator()
{
    num_layers = 0;
    params_set_flag = false;
    all_layer_run_done = false;
}

void Simulator::set_params(Config *config, Topology *topology, char* top_path, bool verbose_flag)
{
    this->config = config;
    this->topology = topology;
    this->top_path = top_path;
    this->verbose = verbose_flag;

    num_layers = this->topology->get_num_layers();

    int64_t word_size = config->get_word_size();
    float active_buf_frac = 0.5;

    auto memSizes = config->get_mem_sizes();
    int64_t ifmap_buf_size_kb = memSizes.ifmap_kb;
    int64_t filter_buf_size_kb = memSizes.filter_kb;
    int64_t ofmap_buf_size_kb = memSizes.ofmap_kb;

    // int64_t ifmap_buf_size_bytes = 1024 * ifmap_buf_size_kb;
    // int64_t filter_buf_size_bytes = 1024 * filter_buf_size_kb;
    // int64_t ofmap_buf_size_bytes = 1024 * ofmap_buf_size_kb;

    int64_t ifmap_buf_size_bytes = ifmap_buf_size_kb;
    int64_t filter_buf_size_bytes = filter_buf_size_kb;
    int64_t ofmap_buf_size_bytes = ofmap_buf_size_kb;

    int64_t bws = config->get_bandwidth();
    int64_t ifmap_backing_bw = bws;
    int64_t filter_backing_bw = bws;
    int64_t ofmap_backing_bw = bws;

    bool prefetch_demand = config->is_prefetch_demand();
    cout << "prefetch_demand " << prefetch_demand << endl;

    int num_pe = config->get_num_pe();

    for (int i = 0; i < num_pe; i++) {
        DoubleBuffer *buffer = new DoubleBuffer();

        if (i == 0) {
            buffer->set_params(
            config,
            word_size,
            ifmap_buf_size_bytes,
            filter_buf_size_bytes,
            ofmap_buf_size_bytes,
            active_buf_frac, active_buf_frac,
            ifmap_backing_bw,
            filter_backing_bw,
            ofmap_backing_bw,
            verbose_flag);
        } else {
            buffer->set_params(
            config,
            word_size,
            ifmap_buf_size_bytes,
            filter_buf_size_bytes,
            ofmap_buf_size_bytes,
            active_buf_frac, active_buf_frac,
            ifmap_backing_bw,
            filter_backing_bw,
            ofmap_backing_bw,
            verbose_flag,
            memory_system[0]->getLLC());
        }        
        memory_system.push_back(buffer);
    }
    params_set_flag = true;
}

void Simulator::run()
{
    // for (int64_t i = 0; i < num_layers; i++)
    // {
    //     LayerSim *layerSim = new LayerSim();
    //     layerSim->set_params(i, config, topology, verbose, memory_system);
    //     single_layer_sim_object_list.push_back(layerSim);
    // }

    if (verbose)
    {
        string name = config->get_run_name();
        name += ".csv";
        ofs = ofstream(name);
        ofs << "readHit,readMissConflict,readMissAll,writeHit,writeMissConflict,writeMissAll" << endl;
    }


    for (int64_t i = 0; i < num_layers; i++)
    {
        LayerSim layerSim;
        layerSim.set_params(i, config, topology, verbose, memory_system);
        // single_layer_sim_object_list.push_back(layerSim);
        if (verbose)
        {
            // int64_t layer_id = single_layer_sim_object_list[i]->get_layer_id();
            int64_t layer_id = layerSim.get_layer_id();
            printf("\nRunning Layer %ld\n", layer_id);
        }
        // single_layer_sim_object_list[i]->run();
        layerSim.run();

        if (verbose) {
            // auto comp_items = single_layer_sim_object_list[i]->get_compute_report_items();
            auto comp_items = layerSim.get_compute_report_items();
            int64_t comp_cycles = comp_items.comp_cycles;
            int64_t stall_cycles = comp_items.stall_cycles;
            float util = comp_items.util;
            float mapping_eff = comp_items.mapping_eff;
            printf("Compute cycles: %ld\n", comp_cycles);
            printf("Stall cycles: %ld\n", stall_cycles);
            printf("Overall utilization: %.2f\n", util);
            printf("Mapping efficiency: %.2f\n", mapping_eff);

            // auto llc_stats = single_layer_sim_object_list[i]->get_llc_stats();
            auto llc_stats = layerSim.get_llc_stats();

            int64_t read_hit = llc_stats.read_hit;
            int64_t read_miss_conflict = llc_stats.read_miss_conflict;
            int64_t read_miss_all = llc_stats.read_miss_all;

            int64_t write_hit = llc_stats.write_hit;
            int64_t write_miss_conflict = llc_stats.write_miss_conflict;
            int64_t write_miss_all = llc_stats.write_miss_all;

            string llc_str = "";
            llc_str += to_string(read_hit);
            llc_str += ",";
            llc_str += to_string(read_miss_conflict);
            llc_str += ",";
            llc_str += to_string(read_miss_all);
            llc_str += ",";
            llc_str += to_string(write_hit);
            llc_str += ",";
            llc_str += to_string(write_miss_conflict);
            llc_str += ",";
            llc_str += to_string(write_miss_all);

            ofs << llc_str << endl;

            // auto avg_bw_items = single_layer_obj.get_bandwidth_report_items();
            // float avg_ifmap_bw = avg_bw_items.avg_ifmap_bw;
            // float avg_filter_bw = avg_bw_items.avg_filter_bw;
            // float avg_ofmap_bw = avg_bw_items.avg_ofmap_bw;

            // printf("Average IFMAP DRAM BW: %.3f words/cycle\n", avg_ifmap_bw);
            // printf("Average Filter DRAM BW: %.3f words/cycle\n", avg_filter_bw);
            // printf("Average OFMAP DRAM BW: %.3f words/cycle\n", avg_ofmap_bw);
        }

        // delete(single_layer_sim_object_list[i]);

        // delete(single_layer_sim_object_list[i]->operandMatrix);
        // delete(single_layer_sim_object_list[i]->compute_system);
        // delete(layerSim);
    }

    all_layer_run_done = true;
    // generate_reports();
    if (verbose) {
        ofs.close();
    }
    
}

void Simulator::generate_reports()
{
    ofstream myfile;
    string file_name = config->get_run_name() + "_shape.csv";
    myfile.open (file_name);
    myfile << "Layer name,ifmap_op_mat_H,ifmap_op_mat_W,filter_op_mat_H,filter_op_mat_W,ofmap_op_mat_H,ofmap_op_mat_W" << endl;
    for (int64_t i = 0; i < num_layers; i++) {
        xt::xarray<int64_t> ifmap_op_mat = single_layer_sim_object_list[i]->getOperandMatrix()->get_ifmap_matrix();
        xt::xarray<int64_t> filter_op_mat = single_layer_sim_object_list[i]->getOperandMatrix()->get_filter_matrix();
        xt::xarray<int64_t> ofmap_op_mat = single_layer_sim_object_list[i]->getOperandMatrix()->get_ofmap_matrix();  

        string layer_name = topology->get_layer_name(i);

        std::string pooling_str ("Pool");

        if (layer_name.find(pooling_str) == std::string::npos) {
            myfile << layer_name;
            myfile << ",";
            myfile << ifmap_op_mat.shape()[0];
            myfile << ",";
            myfile << ifmap_op_mat.shape()[1];
            myfile << ",";
            myfile << filter_op_mat.shape()[0];
            myfile << ",";
            myfile << filter_op_mat.shape()[1];
            myfile << ",";
            myfile << ofmap_op_mat.shape()[0];
            myfile << ",";
            myfile << ofmap_op_mat.shape()[1];
            myfile << endl;
        }

    }
    myfile.close();


    file_name = config->get_run_name() + "_reuse.csv";
    myfile.open (file_name);
    myfile << "group,,IS,OS,WS" << endl;
    for (int64_t i = 0; i < num_layers; i++) {
        xt::xarray<int64_t> ifmap_op_mat = single_layer_sim_object_list[i]->getOperandMatrix()->get_ifmap_matrix();
        xt::xarray<int64_t> filter_op_mat = single_layer_sim_object_list[i]->getOperandMatrix()->get_filter_matrix();
        xt::xarray<int64_t> ofmap_op_mat = single_layer_sim_object_list[i]->getOperandMatrix()->get_ofmap_matrix();  

        string layer_name = topology->get_layer_name(i);

        std::string pooling_str ("Pool");
        std::string fc_str ("FC");
        std::string repeat_str_1 ("b");
        std::string repeat_str_2 ("_2");

        int S = config->get_array_dims().arr_h;

        if (layer_name.find(fc_str) == std::string::npos && layer_name.find(pooling_str) == std::string::npos && layer_name.find(repeat_str_1) == std::string::npos && layer_name.find(repeat_str_2) == std::string::npos) {
            myfile << layer_name;
            myfile << ",";
            myfile << "inputs,";
            int64_t is_reuse = ifmap_op_mat.shape()[0] * ifmap_op_mat.shape()[1];
            int64_t os_reuse = filter_op_mat.shape()[0] * filter_op_mat.shape()[1] * ((ifmap_op_mat.shape()[0] + S - 1)/ S);
            int64_t ws_reuse = ifmap_op_mat.shape()[0] * ifmap_op_mat.shape()[1] * ((ofmap_op_mat.shape()[1] + S - 1)/ S);
            myfile << is_reuse;
            myfile << ",";
            myfile << os_reuse;
            myfile << ",";
            myfile << ws_reuse;
            myfile << endl;

            myfile << layer_name;
            myfile << ",";
            myfile << "weights,";
            is_reuse = filter_op_mat.shape()[0] * filter_op_mat.shape()[1] * ((ifmap_op_mat.shape()[0] + S - 1)/ S);
            os_reuse = filter_op_mat.shape()[0] * filter_op_mat.shape()[1] * ((ifmap_op_mat.shape()[0] + S - 1)/ S);
            ws_reuse = filter_op_mat.shape()[0] * filter_op_mat.shape()[1];
            myfile << is_reuse;
            myfile << ",";
            myfile << os_reuse;
            myfile << ",";
            myfile << ws_reuse;
            myfile << endl;


            myfile << layer_name;
            myfile << ",";
            myfile << "outputs,";
            is_reuse = filter_op_mat.shape()[0] * filter_op_mat.shape()[1] * ((ifmap_op_mat.shape()[0] + S - 1)/ S);
            os_reuse = ofmap_op_mat.shape()[0] * ofmap_op_mat.shape()[1];
            ws_reuse = ifmap_op_mat.shape()[0] * ifmap_op_mat.shape()[1] * ((ofmap_op_mat.shape()[1] + S - 1)/ S);
            myfile << is_reuse;
            myfile << ",";
            myfile << os_reuse;
            myfile << ",";
            myfile << ws_reuse;
            myfile << endl;
        }

    }
    myfile.close();
}

#endif