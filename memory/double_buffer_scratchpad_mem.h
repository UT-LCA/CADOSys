#ifndef _double_buffer_scratchpad_mem_h
#define _double_buffer_scratchpad_mem_h

#include "../scale_config.h"
#include "../topology_utils.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "read_buffer.h"
#include "write_buffer.h"
#include "llc.h"
#include "dram.h"

class DoubleBuffer
{
public:
    DoubleBuffer();
    void set_params(Config* config, 
                             int64_t word_size,
                             int64_t ifmap_buf_size_bytes,
                             int64_t filter_buf_size_bytes,
                             int64_t ofmap_buf_size_bytes,
                             float rd_buf_active_frac, float wr_buf_active_frac,
                             int64_t ifmap_backing_bw,
                             int64_t filter_backing_bw,
                             int64_t ofmap_backing_bw,
                             bool verbose);
    void set_params(Config* config, 
                             int64_t word_size,
                             int64_t ifmap_buf_size_bytes,
                             int64_t filter_buf_size_bytes,
                             int64_t ofmap_buf_size_bytes,
                             float rd_buf_active_frac, float wr_buf_active_frac,
                             int64_t ifmap_backing_bw,
                             int64_t filter_backing_bw,
                             int64_t ofmap_backing_bw,
                             bool verbose,
                             LLC *llc);
    void set_read_buf_prefetch_matrices(xt::xarray<int64_t> ifmap_prefetch_mat, xt::xarray<int64_t> filter_prefetch_mat, xt::xarray<int64_t> ofmap_prefetch_mat);
    void service_memory_requests(xt::xarray<int64_t> &ifmap_demand_mat, xt::xarray<int64_t> &filter_demand_mat, xt::xarray<int64_t> &ofmap_demand_mat, bool trans_ifmap, bool trans_filter, bool trans_ofmap);
    void service_prefetch_demand_memory_requests(xt::xarray<int64_t> ifmap_op_mat, xt::xarray<int64_t> filter_op_mat, 
    xt::xarray<int64_t> ifmap_prefetch_demand_mat, xt::xarray<int64_t> filter_prefetch_demand_mat);
    LLC* getLLC() {return llc;}
    ReadBuffer* get_ifmap_L1_buf() {return ifmap_L1_buf;}
    ReadBuffer* get_filter_L1_buf() {return filter_L1_buf;}

    int64_t get_total_compute_cycles() { return total_cycles; }
    int64_t get_stall_cycles() { return stall_cycles; }

    void add_total_compute_cycles(int64_t cycles) { total_cycles += cycles;}
    void add_stall_cycles(int64_t cycles) { stall_cycles += cycles;}

private:
    Config* config;

    ReadBuffer *ifmap_L1_buf;
    ReadBuffer *filter_L1_buf;
    WriteBuffer *ofmap_L1_buf;

    LLC *llc;
    DRAM *dram;

    bool verbose;

    int64_t total_cycles;
    int64_t compute_cycles;
    int64_t stall_cycles;

    int64_t ifmap_serviced_cycles;
    int64_t filter_serviced_cycles;
    int64_t ofmap_serviced_cycles;

    int64_t avg_ifmap_dram_bw;
    int64_t avg_filter_dram_bw;
    int64_t avg_ofmap_dram_bw;

    int64_t ifmap_sram_start_cycle;
    int64_t ifmap_sram_stop_cycle;
    int64_t filter_sram_start_cycle;
    int64_t filter_sram_stop_cycle;
    int64_t ofmap_sram_start_cycle;
    int64_t ofmap_sram_stop_cycle;

    int64_t ifmap_dram_start_cycle;
    int64_t ifmap_dram_stop_cycle;
    int64_t ifmap_dram_reads;
    int64_t filter_dram_start_cycle;
    int64_t filter_dram_stop_cycle;
    int64_t filter_dram_reads;
    int64_t ofmap_dram_start_cycle;
    int64_t ofmap_dram_stop_cycle;
    int64_t ofmap_dram_writes;

    bool estimate_bandwidth_mode;
    bool traces_valid;
    bool params_valid_flag;

    int arr_row;
    int arr_col;
};

DoubleBuffer::DoubleBuffer() {
    total_cycles = 0;
    stall_cycles = 0;

    ifmap_serviced_cycles = 0;
    filter_serviced_cycles = 0;
    ofmap_serviced_cycles = 0;
}

void DoubleBuffer::set_params(Config* config,
                             int64_t word_size,
                             int64_t ifmap_buf_size_bytes,
                             int64_t filter_buf_size_bytes,
                             int64_t ofmap_buf_size_bytes,
                             float rd_buf_active_frac, float wr_buf_active_frac,
                             int64_t ifmap_backing_bw,
                             int64_t filter_backing_bw,
                             int64_t ofmap_backing_bw,
                             bool verbose)
{
    this->config = config;
    dram = new DRAM();

    llc = new LLC();
    auto llcConfig = config->get_llc_config();
    llc->set_params(dram, llcConfig.total_size_bytes, llcConfig.cache_line_size, 
        llcConfig.hit_latency, llcConfig.set_associativity, llcConfig.partition, llcConfig.is_always_hit, llcConfig.is_bypassing);
        
    ifmap_L1_buf = new ReadBuffer(false);
    filter_L1_buf = new ReadBuffer(false);

    ifmap_L1_buf->set_params(llc, ifmap_buf_size_bytes, word_size, rd_buf_active_frac, ifmap_backing_bw);
    filter_L1_buf->set_params(llc, filter_buf_size_bytes, word_size, rd_buf_active_frac, filter_backing_bw);

    ofmap_L1_buf = new WriteBuffer();
    ofmap_L1_buf->set_params(llc, ofmap_buf_size_bytes, word_size, wr_buf_active_frac, ofmap_backing_bw);

    this->verbose = verbose;
    params_valid_flag = true;
}

void DoubleBuffer::set_params(Config* config,
                             int64_t word_size,
                             int64_t ifmap_buf_size_bytes,
                             int64_t filter_buf_size_bytes,
                             int64_t ofmap_buf_size_bytes,
                             float rd_buf_active_frac, float wr_buf_active_frac,
                             int64_t ifmap_backing_bw,
                             int64_t filter_backing_bw,
                             int64_t ofmap_backing_bw,
                             bool verbose,
                             LLC *llc)
{
    this->config = config;
    dram = new DRAM();

    this->llc = llc;

    ifmap_L1_buf = new ReadBuffer(false);
    filter_L1_buf = new ReadBuffer(false);

    ifmap_L1_buf->set_params(llc, ifmap_buf_size_bytes, word_size, rd_buf_active_frac, ifmap_backing_bw);
    filter_L1_buf->set_params(llc, filter_buf_size_bytes, word_size, rd_buf_active_frac, filter_backing_bw);

    ofmap_L1_buf = new WriteBuffer();
    ofmap_L1_buf->set_params(llc, ofmap_buf_size_bytes, word_size, wr_buf_active_frac, ofmap_backing_bw);

    this->verbose = verbose;
    params_valid_flag = true;
}

void DoubleBuffer::set_read_buf_prefetch_matrices(xt::xarray<int64_t> ifmap_prefetch_mat, xt::xarray<int64_t> filter_prefetch_mat, xt::xarray<int64_t> ofmap_prefetch_mat) {
    ifmap_L1_buf->set_fetch_matrix(ifmap_prefetch_mat);
    filter_L1_buf->set_fetch_matrix(filter_prefetch_mat);
    ofmap_L1_buf->set_fetch_matrix(ofmap_prefetch_mat);
}


void DoubleBuffer::service_memory_requests(xt::xarray<int64_t> &ifmap_demand_mat, xt::xarray<int64_t> &filter_demand_mat, xt::xarray<int64_t> &ofmap_demand_mat, bool trans_ifmap, bool trans_filter, bool trans_ofmap) {
    int64_t ofmap_lines = ofmap_demand_mat.shape()[0];

    int64_t ifmap_hit_latency = ifmap_L1_buf->get_hit_latency();
    int64_t filter_hit_latency = ifmap_L1_buf->get_hit_latency();

    int64_t current_stall_cycles = 0;

    for (int64_t i = 0; i < ofmap_lines; i++) {
        // cout << "process " << i << " of " << ofmap_lines << endl;
        int64_t incoming_cycle_arr = 1 + i + current_stall_cycles;

        xt::xarray<int64_t> ifmap_demand_line = xt::view(ifmap_demand_mat, i, xt::all());
        int64_t ifmap_cycle_out;
        if (config->is_use_llc_partition()) {
            // ifmap_cycle_out = ifmap_L1_buf->service_read(ifmap_demand_line, incoming_cycle_arr, 0, trans_ifmap);
            ifmap_cycle_out = ifmap_L1_buf->service_read(i, incoming_cycle_arr, 0, trans_ifmap);
        } else {
            // ifmap_cycle_out = ifmap_L1_buf->service_read(ifmap_demand_line, incoming_cycle_arr, 0, trans_ifmap);
            ifmap_cycle_out = ifmap_L1_buf->service_read(i, incoming_cycle_arr, 0, trans_ifmap);
        }
        ifmap_serviced_cycles = ifmap_cycle_out;
        int64_t ifmap_stalls = ifmap_cycle_out - incoming_cycle_arr - ifmap_hit_latency;

        // cout << "ifmap_demand_line is " << ifmap_demand_line << endl;
        // cout << "ifmap_serviced_cycles is " << ifmap_serviced_cycles << endl;
            
        xt::xarray<int64_t> filter_demand_line = xt::view(filter_demand_mat, i, xt::all());
        int64_t filter_cycle_out;
        if (config->is_use_llc_partition()) {
            // filter_cycle_out = filter_L1_buf->service_read(filter_demand_line, incoming_cycle_arr, 1, trans_filter);
            filter_cycle_out = filter_L1_buf->service_read(i, incoming_cycle_arr, 1, trans_filter);
        } else {
            // filter_cycle_out = filter_L1_buf->service_read(filter_demand_line, incoming_cycle_arr, 0, trans_filter);
            filter_cycle_out = filter_L1_buf->service_read(i, incoming_cycle_arr, 0, trans_filter);
        }
        filter_serviced_cycles = filter_cycle_out;
        int64_t filter_stalls = filter_cycle_out - incoming_cycle_arr - filter_hit_latency;

        // cout << "filter_demand_line is " << filter_demand_line << endl;
        // cout << "filter_serviced_cycles is " << filter_serviced_cycles << endl;

        xt::xarray<int64_t> ofmap_demand_line = xt::view(ofmap_demand_mat, i, xt::all());
        int64_t ofmap_cycle_out;
        if (config->is_use_llc_partition()) {
            // ofmap_cycle_out = ofmap_L1_buf->service_write(ofmap_demand_line, incoming_cycle_arr, 0, trans_ofmap);
            ofmap_cycle_out = ofmap_L1_buf->service_write(i, incoming_cycle_arr, 0, trans_ofmap);
        } else {
            // ofmap_cycle_out = ofmap_L1_buf->service_write(ofmap_demand_line, incoming_cycle_arr, 0, trans_ofmap);
            ofmap_cycle_out = ofmap_L1_buf->service_write(i, incoming_cycle_arr, 0, trans_ofmap);
        }
        ofmap_serviced_cycles = ofmap_cycle_out;
        int64_t ofmap_stalls = ofmap_cycle_out - incoming_cycle_arr - 1;

        // cout << "ofmap_demand_line is " << ofmap_demand_line << endl;
        // cout << "ofmap_serviced_cycles is " << ofmap_serviced_cycles << endl;
        current_stall_cycles += max(ifmap_stalls, max(filter_stalls, ofmap_stalls));
    }
    llc->dump_stats();

    cout << "current_stall_cycles is " << current_stall_cycles << endl;
    cout << "ifmap_serviced_cycles is " << ifmap_serviced_cycles << endl;
    cout << "filter_serviced_cycles is " << filter_serviced_cycles << endl;
    cout << "ofmap_serviced_cycles is " << ofmap_serviced_cycles << endl;

    stall_cycles += current_stall_cycles;
    total_cycles += ofmap_serviced_cycles;
}


void DoubleBuffer::service_prefetch_demand_memory_requests(xt::xarray<int64_t> ifmap_op_mat, xt::xarray<int64_t> filter_op_mat, 
    xt::xarray<int64_t> ifmap_prefetch_demand_mat, xt::xarray<int64_t> filter_prefetch_demand_mat) {

}

#endif