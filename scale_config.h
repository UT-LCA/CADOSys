#ifndef _scale_config_h
#define _scale_config_h

#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ini_parser.hpp>
using namespace boost;
using namespace std;

#include "memory_map.h"

typedef struct {
    int64_t arr_h;
    int64_t arr_w;
} ArrayDims;

typedef struct {
    int64_t ifmap_kb;
    int64_t filter_kb;
    int64_t ofmap_kb;
} MemSizes;

typedef struct {
    int64_t ifmap_offset;
    int64_t filter_offset;
    int64_t ofmap_offset;
} MemOffsets;

typedef struct {
    int64_t total_size_bytes;
    int64_t cache_line_size;
    int64_t hit_latency;
    int64_t set_associativity;
    string partition;
    bool is_always_hit;
    bool is_bypassing;
} LlcConfig;

class Config
{
public:
    Config();
    void read_conf_file(char *conf_file_in);
    ArrayDims get_array_dims() { return arrayDims; }
    MemSizes get_mem_sizes() { return memSizes; }
    MemOffsets get_mem_offsets() { return memOffsets; }
    LlcConfig get_llc_config() { return llcConfig; }
    
    string get_run_name() {return run_name; }
    string get_dataflow() {return df;}
    int64_t get_unified() {return unified;}
    int64_t get_mem_banks() {return memory_banks;}
    int64_t get_bandwidth() {return bandwidth;}
    string get_topology_path() {return topofile;}
    int64_t get_batch_size() {return batch_size;}
    int64_t get_word_size() {return word_size;}
    bool is_prefetch_demand() {return prefetch_demand;}
    bool is_use_llc_partition() {return use_llc_partition; }
    int get_num_pe() {return num_pe; }
    bool is_tensor_main_order() {return tensor_main_order; }

private:
    string run_name;

    ArrayDims arrayDims;
    MemSizes memSizes;
    MemOffsets memOffsets;
    LlcConfig llcConfig;

    string df;
    int64_t unified;

    string topofile;
    int64_t bandwidth;
    int64_t memory_banks;
    int64_t word_size;
    int64_t batch_size;
    bool prefetch_demand;
    bool use_llc_partition;
    int num_pe;
    bool tensor_main_order;

    int llc_size;
    int llc_assoc;
    int llc_partition;

    MemoryMap *memory_map;

    bool valid_conf_flag = false;

    string valid_df_list[3] = {"os", "ws", "is"};
};

Config::Config()
{
    run_name = "scale_run";

    arrayDims.arr_h = 4;
    arrayDims.arr_w = 4;

    memSizes.ifmap_kb = 256;
    memSizes.filter_kb = 256;
    memSizes.ofmap_kb = 128;

    df = "ws";
    unified = 1;

    memOffsets.ifmap_offset = 0;
    memOffsets.filter_offset = 10000000;
    memOffsets.ofmap_offset = 20000000;

    topofile = "";
    bandwidth = 32;
    memory_banks = 1;
    word_size = 4;
    batch_size = 1;
    num_pe = 1;

    llcConfig.total_size_bytes = 1 * 1024 * 1024;
    llcConfig.cache_line_size = 64;
    llcConfig.hit_latency = 1;
    llcConfig.set_associativity = 4;
    llcConfig.partition = "16";

    memory_map = new MemoryMap();

    valid_conf_flag = false;
}

void Config::read_conf_file(char *conf_file_in)
{
    topofile = conf_file_in;
    property_tree::ptree m_data;
    property_tree::ini_parser::read_ini(topofile, m_data);

    run_name = m_data.get<string>("general.run_name");

    arrayDims.arr_h = m_data.get<int64_t>("architecture_presets.ArrayHeight");
    arrayDims.arr_w = m_data.get<int64_t>("architecture_presets.ArrayWidth");

    memSizes.ifmap_kb = m_data.get<int64_t>("architecture_presets.IfmapSramSzkB");
    memSizes.filter_kb = m_data.get<int64_t>("architecture_presets.FilterSramSzkB");
    memSizes.ofmap_kb = m_data.get<int64_t>("architecture_presets.OfmapSramSzkB");

    memOffsets.ifmap_offset = m_data.get<int64_t>("architecture_presets.IfmapOffset");
    memOffsets.filter_offset = m_data.get<int64_t>("architecture_presets.FilterOffset");
    memOffsets.ofmap_offset = m_data.get<int64_t>("architecture_presets.OfmapOffset");

    df = m_data.get<string>("architecture_presets.Dataflow");
    unified = m_data.get<int64_t>("architecture_presets.Unified");

    memory_banks = m_data.get<int64_t>("architecture_presets.MemoryBanks");
    word_size = m_data.get<int64_t>("architecture_presets.WordSize");
    batch_size = m_data.get<int64_t>("architecture_presets.BatchSize");

    bandwidth = m_data.get<int64_t>("architecture_presets.Bandwidth");
    prefetch_demand = m_data.get<bool>("architecture_presets.PrefetchDemand");
    use_llc_partition = m_data.get<bool>("architecture_presets.UseLLCPartition"); 
    num_pe = m_data.get<int>("architecture_presets.NumPE"); 
    tensor_main_order = m_data.get<bool>("architecture_presets.TensorMainOrder"); 

    llcConfig.total_size_bytes = m_data.get<int64_t>("llc.SizekB") * 1024;
    llcConfig.cache_line_size = m_data.get<int64_t>("llc.CacheLineSize");
    llcConfig.hit_latency = m_data.get<int64_t>("llc.HitLatency");
    llcConfig.set_associativity = m_data.get<int64_t>("llc.Assoc");
    llcConfig.partition = m_data.get<string>("llc.Partition");
    llcConfig.is_always_hit = m_data.get<bool>("llc.AlwaysHit");
    llcConfig.is_bypassing = m_data.get<bool>("llc.Bypassing");

    memory_map->set_single_bank_params(memOffsets.filter_offset, memOffsets.ofmap_offset);
}


#endif