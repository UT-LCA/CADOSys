#ifndef _layer_sim_h
#define _layer_sim_h

#include <string>
#include <iostream>
#include <vector>

#include "scale_config.h"
#include "topology_utils.h"

#include "compute/operand_matrix.h"
#include "compute/systolic_compute.h"
#include "compute/systolic_compute_os.h"
#include "compute/systolic_compute_is.h"
#include "compute/systolic_compute_ws.h"
#include "compute/systolic_pool_ws.h"
#include "compute/systolic_pool_os.h"
#include "memory/double_buffer_scratchpad_mem.h"

typedef struct
{
    int64_t comp_cycles;
    int64_t stall_cycles;
    float util;
    float mapping_eff;
} ComputeStats;

typedef struct
{
    float avg_ifmap_bw;
    float avg_filter_bw;
    float avg_ofmap_bw;
} BandwidthStats;

class LayerSim
{
public:
    LayerSim();
    // ~LayerSim();
    void set_params(int64_t layer_id, Config *config, Topology *topology, bool verbose, vector<DoubleBuffer*> memory_system);
    int64_t get_layer_id() { return layer_id; }
    void run();

    ComputeStats get_compute_report_items();
    BandwidthStats get_bandwidth_report_items();
    LLCStats get_llc_stats() {return memory_system[0]->getLLC()->get_llc_stats();}
    OperandMatrix* getOperandMatrix() {return operandMatrix;}

private:
    int64_t layer_id;
    vector<int> pe_list;
    Config *config;
    Topology *topology;
    bool verbose;

    string dataflow;

    OperandMatrix *operandMatrix;
    SystolicCompute *compute_system;
    vector<DoubleBuffer*> memory_system;

    xt::xarray<int64_t> ifmap_op_mat;
    xt::xarray<int64_t> filter_op_mat;
    xt::xarray<int64_t> ofmap_op_mat;

    int64_t total_cycles;
    int64_t stall_cycles;
    int64_t num_compute;
    int64_t num_mac_unit;
    float overall_util;
    float mapping_eff;
    float compute_util;

    bool params_set_flag = false;
    bool runs_ready = false;
    bool report_items_ready = false;

    void calc_report_data();
};

LayerSim::LayerSim()
{
    operandMatrix = new OperandMatrix();
}

// LayerSim::~LayerSim()
// {
//     delete(operandMatrix);
//     int64_t layer_type = topology->get_layer_type(this->layer_id);
//     if (layer_type == CONV) {
//         if (dataflow == "os") {
//             delete((SystolicComputeOs *) compute_system);
//         } else if (dataflow == "is") {
//             delete((SystolicComputeIs *) compute_system);
//         } else if (dataflow == "ws") {
//             delete((SystolicComputeWs *) compute_system);
//         } else {
//             delete((SystolicComputeWs *) compute_system);
//         }
//     } else if (layer_type == POOL) {
//         delete((SystolicPoolOs *) compute_system);
//     }
//     // delete(compute_system);

//     cout << "ifmap_op_mat.data() is " << reinterpret_cast<void *>(ifmap_op_mat.data()) << endl;
//     cout << "ifmap_op_mat.size() is " << ifmap_op_mat.size() << endl;

//     ifmap_op_mat = xt::ones<int64_t>({1, 1});
    
//     cout << "ifmap_op_mat.data() is " << reinterpret_cast<void *>(ifmap_op_mat.data()) << endl;
//     cout << "ifmap_op_mat.size() is " << ifmap_op_mat.size() << endl;


//     // delete (ifmap_op_mat.data());
//     // delete [] ifmap_op_mat.data();
//     // delete [] ifmap_op_mat;
//     // delete(filter_op_mat.data());
//     // delete(ofmap_op_mat.data());
//     // ifmap_op_mat.resize({1, 1}, true);
//     // ifmap_op_mat = xt::ones<int64_t>({1, 1});
//     // filter_op_mat = xt::ones<int64_t>({1, 1});
//     // ofmap_op_mat = xt::ones<int64_t>({1, 1});
// }

void LayerSim::set_params(int64_t layer_id, Config *config, Topology *topology, bool verbose, vector<DoubleBuffer*> memory_system)
{
    this->layer_id = layer_id;
    this->config = config;
    this->topology = topology;
    this->verbose = verbose;
    this->memory_system = memory_system;

    operandMatrix->set_params(config, topology, layer_id);

    // this->dataflow = this->config->get_dataflow();
    this->dataflow = topology->get_layer_dataflow(this->layer_id);

    int64_t layer_type = topology->get_layer_type(this->layer_id);

    if (layer_type == CONV) {
        if (dataflow == "os") {
            this->compute_system = (SystolicComputeOs *)new SystolicComputeOs();
        } else if (dataflow == "is") {
            this->compute_system = (SystolicComputeIs *)new SystolicComputeIs();
        } else if (dataflow == "ws") {
            this->compute_system = (SystolicComputeWs *)new SystolicComputeWs();
        } else {
            this->compute_system = (SystolicComputeWs *)new SystolicComputeWs();
        }
    } else if (layer_type == POOL) {
        this->compute_system = (SystolicPoolOs *)new SystolicPoolOs();
    }

    

    auto arr_dims = this->config->get_array_dims();
    this->num_mac_unit = arr_dims.arr_h * arr_dims.arr_w;

    pe_list = topology->get_layer_pe_list(this->layer_id);
    params_set_flag = true;
}

void LayerSim::run()
{
    ifmap_op_mat = operandMatrix->get_ifmap_matrix();
    filter_op_mat = operandMatrix->get_filter_matrix();
    ofmap_op_mat = operandMatrix->get_ofmap_matrix();    

    num_compute = topology->get_layer_num_ofmap_px(this->layer_id) * topology->get_layer_window_size(this->layer_id);

    this->compute_system->set_params(config, ifmap_op_mat, filter_op_mat, ofmap_op_mat, pe_list.size());

    // xt::xarray<int64_t> ifmap_prefetch_mat = this->compute_system->get_ifmap_prefetch_matrices();
    // xt::xarray<int64_t> filter_prefetch_mat = this->compute_system->get_filter_prefetch_matrices();

    vector<xt::xarray<int64_t>> ifmap_demand_mats = this->compute_system->get_ifmap_demand_matrices();
    vector<xt::xarray<int64_t>> filter_demand_mats = this->compute_system->get_filter_demand_matrices();
    vector<xt::xarray<int64_t>> ofmap_demand_mats = this->compute_system->get_ofmap_demand_matrices();

    int64_t layer_type = topology->get_layer_type(this->layer_id);

    for (int i = 0; i < pe_list.size(); i++) {
        memory_system[pe_list[i]]->set_read_buf_prefetch_matrices(ifmap_demand_mats[i], filter_demand_mats[i], ofmap_demand_mats[i]);
        if (layer_type == CONV) {
            if (dataflow == "os") {
                memory_system[pe_list[i]]->service_memory_requests(ifmap_demand_mats[i], filter_demand_mats[i], ofmap_demand_mats[i], 1, 1, 0);
            } else if (dataflow == "is") {
                memory_system[pe_list[i]]->service_memory_requests(ifmap_demand_mats[i], filter_demand_mats[i], ofmap_demand_mats[i], 1, 0, 1);
            } else {
                memory_system[pe_list[i]]->service_memory_requests(ifmap_demand_mats[i], filter_demand_mats[i], ofmap_demand_mats[i], 0, 0, 0);
            }   
        } else if (layer_type == POOL) {
            memory_system[pe_list[i]]->service_memory_requests(ifmap_demand_mats[i], filter_demand_mats[i], ofmap_demand_mats[i], 1, 1, 0);
        }
             
    }

    // delete(&ifmap_op_mat);
    // delete(&filter_demand_mats);
    // delete(&ofmap_demand_mats);

    ifmap_demand_mats.clear();
    filter_demand_mats.clear();
    ofmap_demand_mats.clear();

    // delete(operandMatrix);
    // delete(compute_system);

    runs_ready = true;
}

void LayerSim::calc_report_data()
{
    total_cycles = 0;
    stall_cycles = 0;
    for (int i = 0; i < pe_list.size(); i++) {
        total_cycles += memory_system[pe_list[i]]->get_total_compute_cycles();
        stall_cycles += memory_system[pe_list[i]]->get_stall_cycles();
    }
    overall_util = (num_compute * 100) / (total_cycles * num_mac_unit);
    mapping_eff = compute_system->get_avg_mapping_efficiency() * 100;
    compute_util = compute_system->get_avg_compute_utilization() * 100;

    report_items_ready = true;
}

ComputeStats LayerSim::get_compute_report_items()
{
    if (!report_items_ready)
        calc_report_data();

    ComputeStats computeStats;
    computeStats.comp_cycles = total_cycles;
    computeStats.stall_cycles = stall_cycles;
    computeStats.util = overall_util;
    computeStats.mapping_eff = mapping_eff;
    return computeStats;
}

BandwidthStats LayerSim::get_bandwidth_report_items()
{
    if (!report_items_ready)
        calc_report_data();

    BandwidthStats bandwidthStats;
    bandwidthStats.avg_ifmap_bw = 0.0f;
    bandwidthStats.avg_filter_bw = 0.0f;
    bandwidthStats.avg_ofmap_bw = 0.0f;
    return bandwidthStats;
}

#endif