#ifndef _topology_utils_h
#define _topology_utils_h

#include <vector>
#include <string>
#include "csv.h"

#include "scale_config.h"

#define CONV 0
#define POOL 1

using namespace std;

typedef struct
{
    string name;
    string dataflow;

    int64_t type;

    int64_t ifmap_height;
    int64_t ifmap_width;

    int64_t filter_height;
    int64_t filter_width;

    int64_t channels;
    int64_t num_filer;

    int64_t stride_height;
    int64_t stride_width;

    vector<int64_t> ifmap_offset;
    int64_t filter_offset;
    int64_t ofmap_offset;

    int64_t ifmap_demand_offset;
    int64_t filter_demand_offset;

    int64_t ifmap_offset_end;
    int64_t filter_offset_end;
    int64_t ofmap_offset_end;
    vector<int> pe_list;
} LayerInfo;

typedef struct
{
    int64_t ofmap_height;
    int64_t ofmap_width;
    int64_t num_mac;
    int64_t window_size;
} CalcLayerInfo;

typedef struct
{
    vector<int64_t> ifmap_offset;
    int64_t filter_offset;
    int64_t ofmap_offset;
} OffsetInfo;

// Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Stride Height, Stride Width, IFMAP Offset, Filter Offset, OFMAP Offset,
// Conv1, 224, 224, 11, 11, 3, 96, 4, 4, 0, 10000000, 20000000,

class Topology
{
public:
    Topology();
    void load_arrays(Config *config, char *topofile, bool is_prefetch, bool mnk_inputs);
    int64_t get_num_layers() { return num_layers; }

    pair<int64_t, int64_t> get_layer_ifmap_dims(int64_t layer_id) { return pair<int64_t, int64_t>(topo_arrays[layer_id].ifmap_height, topo_arrays[layer_id].ifmap_width); }
    pair<int64_t, int64_t> get_layer_filter_dims(int64_t layer_id) { return pair<int64_t, int64_t>(topo_arrays[layer_id].filter_height, topo_arrays[layer_id].filter_width); }

    int64_t get_layer_num_channels(int64_t layer_id) { return topo_arrays[layer_id].channels; }
    int64_t get_layer_num_filters(int64_t layer_id) { return topo_arrays[layer_id].num_filer; }
    pair<int64_t, int64_t> get_layer_strides(int64_t layer_id) { return pair<int64_t, int64_t>(topo_arrays[layer_id].stride_height, topo_arrays[layer_id].stride_width); }

    OffsetInfo get_layer_offsets(int64_t layer_id);

    pair<int64_t, int64_t> get_layer_ofmap_dims(int64_t layer_id) { return pair<int64_t, int64_t>(calc_topo_arrays[layer_id].ofmap_height, calc_topo_arrays[layer_id].ofmap_width); }
    int64_t get_layer_window_size(int64_t layer_id) { return calc_topo_arrays[layer_id].window_size; }
    int64_t get_layer_num_ofmap_px(int64_t layer_id);

    int64_t get_layer_ifmap_demand_offset(int64_t layer_id) { return topo_arrays[layer_id].ifmap_demand_offset; }
    int64_t get_layer_filter_demand_offset(int64_t layer_id) { return topo_arrays[layer_id].filter_demand_offset; }
    int64_t get_layer_type(int64_t layer_id) { return topo_arrays[layer_id].type; }
    string get_layer_name(int64_t layer_id) { return topo_arrays[layer_id].name; }
    string get_layer_dataflow(int64_t layer_id) {return topo_arrays[layer_id].dataflow;}
    vector<int> get_layer_pe_list(int64_t layer_id) { return topo_arrays[layer_id].pe_list; }

private:
    Config *config;
    string current_topo_name;
    string topo_file_name;

    vector<LayerInfo> topo_arrays;
    vector<CalcLayerInfo> calc_topo_arrays;

    int64_t num_layers;
    bool topo_load_flag = false;
    bool topo_calc_hyper_param_flag = false;
    bool topo_calc_spatiotemp_params_flag = false;

    void load_arrays_gemm(char *topofile);
    void load_arrays_conv(char *topofile, bool is_prefetch);
};

Topology::Topology()
{
    current_topo_name = "";
    topo_file_name = "";

    num_layers = 0;

    topo_load_flag = false;
    topo_calc_hyper_param_flag = false;
    topo_calc_spatiotemp_params_flag = false;
}

void Topology::load_arrays(Config *config, char *topofile, bool is_prefetch_demand, bool mnk_inputs = false)
{
    this->config = config;
    if (mnk_inputs)
    {
        load_arrays_gemm(topofile);
    }
    else
    {
        load_arrays_conv(topofile, is_prefetch_demand);
    }
}

void Topology::load_arrays_gemm(char *topofile)
{
    return;
}

void Topology::load_arrays_conv(char *topofile, bool is_prefetch_demand)
{
    auto memoffset = config->get_mem_offsets();

    int64_t initial_ifmap_offset = memoffset.ifmap_offset;
    int64_t initial_filter_offset = memoffset.filter_offset;
    int64_t initial_ofmap_offset = memoffset.ofmap_offset;

    int64_t word_size = config->get_word_size();
    int64_t batch_size = config->get_batch_size();

    bool is_tensor_main_order = config->is_tensor_main_order();

    string name;
    int64_t type;
    int64_t ifmap_height;
    int64_t ifmap_width;
    int64_t filter_height;
    int64_t filter_width;
    int64_t channels;
    int64_t num_filer;
    int64_t stride_height;
    int64_t stride_width;
    string dataflow;

    string ifmap_sources;
    string filter_sources;
    string pes;

    int64_t filter_offset;
    int64_t ifmap_offset;
    int64_t ofmap_offset;

    int64_t ifmap_demand_offset;
    int64_t filter_demand_offset;

    int64_t filter_offset_end;
    int64_t ifmap_offset_end;
    int64_t ofmap_offset_end;

    int64_t ifmap_demand_offset_end;
    int64_t filter_demand_offset_end;

    int64_t unified = config->get_unified();

    if (!unified) {
        csv::CSVReader<14> in(topofile);
        in.read_header(csv::ignore_extra_column, "Layer name", "Layer Type", "IFMAP Height", "IFMAP Width", "Filter Height", "Filter Width", "Channels",
                    "Num Filter", "Stride Height", "Stride Width", "IFMAP Source", "Filter Source", "PE", "Dataflow");

        while (in.read_row(name, type, ifmap_height, ifmap_width, filter_height, filter_width, channels, num_filer, stride_height, stride_width, ifmap_sources, filter_sources, pes, dataflow))
        {
            LayerInfo info;

            info.name = name;
            info.type = type;
            info.ifmap_height = ifmap_height;
            info.ifmap_width = ifmap_width;
            info.filter_height = filter_height;
            info.filter_width = filter_width;
            info.channels = channels;
            info.num_filer = num_filer;
            info.stride_height = stride_height;
            info.stride_width = stride_height;
            info.dataflow = dataflow;

            CalcLayerInfo calcinfo;
            // calcinfo.ofmap_height = (info.ifmap_height - info.filter_height + info.stride_height + info.stride_height - 1) / info.stride_height;
            // calcinfo.ofmap_width = (info.ifmap_width - info.filter_width + info.stride_width + info.stride_width - 1) / info.stride_width;

            calcinfo.ofmap_height = info.ifmap_height / info.stride_height;
            calcinfo.ofmap_width = info.ifmap_width / info.stride_width;

            calcinfo.num_mac = calcinfo.ofmap_height * calcinfo.ofmap_width * info.filter_height * info.filter_width * info.channels * info.num_filer;
            calcinfo.window_size = info.filter_height * info.filter_width * info.channels;
            calc_topo_arrays.push_back(calcinfo);


            vector<string> ifmap_source_str_list;
            vector<int> ifmap_source_list;
            stringstream str_ifmap_source(ifmap_sources);

            while (str_ifmap_source.good())
            {
                string substr;
                getline(str_ifmap_source, substr, '_');
                ifmap_source_str_list.push_back(substr);
            }

            for (size_t i = 0; i < ifmap_source_str_list.size(); i++)
            {
                ifmap_source_list.push_back(stoi(ifmap_source_str_list[i]));
            }


            vector<string> filter_source_str_list;
            vector<int> filter_source_list;
            stringstream str_filter_source(filter_sources);

            cout << "filter_sources is " << filter_sources << endl;

            while (str_filter_source.good())
            {
                string substr;
                getline(str_filter_source, substr, '_');
                filter_source_str_list.push_back(substr);
            }

            for (size_t i = 0; i < filter_source_str_list.size(); i++)
            {
                filter_source_list.push_back(stoi(filter_source_str_list[i]));
            }
            
            vector<string> pe_str_list;
            vector<int> pe_list;
            stringstream str_pe(pes);

            while (str_pe.good())
            {
                string substr;
                getline(str_pe, substr, '_');
                pe_str_list.push_back(substr);
            }

            for (size_t i = 0; i < pe_str_list.size(); i++)
            {
                pe_list.push_back(stoi(pe_str_list[i]));
            }

            info.pe_list = pe_list;


            vector<int64_t> ifmap_offset;

            uint64_t ifmap_size = ifmap_height * ifmap_width * channels * word_size * batch_size * ifmap_source_list.size();
            uint64_t filter_size = 0;
            if (type == CONV)
                filter_size = filter_height * filter_width * channels * num_filer * word_size;
            uint64_t ofmap_size = calcinfo.ofmap_height * calcinfo.ofmap_width * num_filer * word_size * batch_size;

            uint64_t ifmap_demand_size = calcinfo.ofmap_height * calcinfo.ofmap_width * num_filer * filter_height * filter_width * channels * word_size * batch_size;
            uint64_t filter_demand_size = filter_height * filter_width * channels * num_filer * word_size * batch_size;

            if (is_tensor_main_order) {
                if (topo_arrays.size() == 0) {
                    filter_offset = initial_filter_offset;
                    ofmap_offset = initial_ifmap_offset + ifmap_size;
                } else {
                    filter_offset = topo_arrays[topo_arrays.size() - 1].filter_offset_end;
                    ofmap_offset = topo_arrays[topo_arrays.size() - 1].ofmap_offset_end;
                }

                if (ifmap_source_list[0] == -1) {
                    ifmap_offset.push_back(initial_ifmap_offset);
                    initial_ifmap_offset += ifmap_size;
                } else {
                    for (size_t i = 0; i < ifmap_source_list.size(); i++)
                    {
                        ifmap_offset.push_back(topo_arrays[ifmap_source_list[i]].ofmap_offset);
                    }
                }

                if (filter_source_list[0] != -1) {
                    filter_offset = topo_arrays[filter_source_list[0]].ofmap_offset;
                    filter_size = 0;
                }

                info.ifmap_offset = ifmap_offset;
                info.filter_offset = filter_offset;
                
                if (is_prefetch_demand) {
                    info.filter_demand_offset = info.filter_offset + filter_size;
                    info.filter_offset_end = info.filter_demand_offset + filter_demand_size;

                    info.ofmap_offset = ofmap_offset + ifmap_demand_size;
                    info.ofmap_offset_end = info.ofmap_offset + ofmap_size;
                } else {
                    info.filter_demand_offset = info.filter_offset;
                    info.filter_offset_end = info.filter_offset + filter_size;

                    info.ofmap_offset = ofmap_offset;
                    info.ofmap_offset_end = info.ofmap_offset + ofmap_size;
                }
            } else {
                if (ifmap_source_list[0] == -1) {
                    ifmap_offset.push_back(initial_ifmap_offset);
                    initial_ifmap_offset += ifmap_size;
                } else {
                    for (size_t i = 0; i < ifmap_source_list.size(); i++)
                    {
                        ifmap_offset.push_back(topo_arrays[ifmap_source_list[i]].ofmap_offset);
                    }
                }

                if (filter_source_list[0] != -1) {
                    filter_offset = topo_arrays[filter_source_list[0]].ofmap_offset;
                    ofmap_offset = filter_offset + filter_size;
                    filter_size = 0;
                } else {
                    filter_offset = initial_ifmap_offset + ifmap_size;
                    ofmap_offset = filter_offset + filter_size;
                }
                initial_ifmap_offset += filter_size;

                info.ifmap_offset = ifmap_offset;
                info.filter_offset = filter_offset;
                
                if (is_prefetch_demand) {
                    info.filter_demand_offset = info.filter_offset + filter_size;
                    info.filter_offset_end = info.filter_demand_offset + filter_demand_size;

                    info.ofmap_offset = ofmap_offset + ifmap_demand_size;
                    info.ofmap_offset_end = info.ofmap_offset + ofmap_size;
                } else {
                    info.filter_demand_offset = info.filter_offset;
                    info.filter_offset_end = info.filter_offset + filter_size;

                    info.ofmap_offset = ofmap_offset;
                    info.ofmap_offset_end = info.ofmap_offset + ofmap_size;
                }
                initial_ifmap_offset += ofmap_size;
            }
            

            topo_arrays.push_back(info);
        }
    } else {
        csv::CSVReader<13> in(topofile);
        in.read_header(csv::ignore_extra_column, "Layer name", "Layer Type", "IFMAP Height", "IFMAP Width", "Filter Height", "Filter Width", "Channels",
                    "Num Filter", "Stride Height", "Stride Width", "IFMAP Source", "Filter Source", "PE");

        while (in.read_row(name, type, ifmap_height, ifmap_width, filter_height, filter_width, channels, num_filer, stride_height, stride_width, ifmap_sources, filter_sources, pes))
        {
            LayerInfo info;

            info.name = name;
            info.type = type;
            info.ifmap_height = ifmap_height;
            info.ifmap_width = ifmap_width;
            info.filter_height = filter_height;
            info.filter_width = filter_width;
            info.channels = channels;
            info.num_filer = num_filer;
            info.stride_height = stride_height;
            info.stride_width = stride_height;
            info.dataflow = config->get_dataflow();

            CalcLayerInfo calcinfo;

            calcinfo.ofmap_height = info.ifmap_height / info.stride_height;
            calcinfo.ofmap_width = info.ifmap_width / info.stride_width;

            calcinfo.num_mac = calcinfo.ofmap_height * calcinfo.ofmap_width * info.filter_height * info.filter_width * info.channels * info.num_filer;
            calcinfo.window_size = info.filter_height * info.filter_width * info.channels;
            calc_topo_arrays.push_back(calcinfo);


            vector<string> ifmap_source_str_list;
            vector<int> ifmap_source_list;
            stringstream str_ifmap_source(ifmap_sources);

            while (str_ifmap_source.good())
            {
                string substr;
                getline(str_ifmap_source, substr, '_');
                ifmap_source_str_list.push_back(substr);
            }

            for (size_t i = 0; i < ifmap_source_str_list.size(); i++)
            {
                ifmap_source_list.push_back(stoi(ifmap_source_str_list[i]));
            }


            vector<string> filter_source_str_list;
            vector<int> filter_source_list;
            stringstream str_filter_source(filter_sources);

            cout << "filter_sources is " << filter_sources << endl;

            while (str_filter_source.good())
            {
                string substr;
                getline(str_filter_source, substr, '_');
                filter_source_str_list.push_back(substr);
            }

            for (size_t i = 0; i < filter_source_str_list.size(); i++)
            {
                filter_source_list.push_back(stoi(filter_source_str_list[i]));
            }
            
            vector<string> pe_str_list;
            vector<int> pe_list;
            stringstream str_pe(pes);

            while (str_pe.good())
            {
                string substr;
                getline(str_pe, substr, '_');
                pe_str_list.push_back(substr);
            }

            for (size_t i = 0; i < pe_str_list.size(); i++)
            {
                pe_list.push_back(stoi(pe_str_list[i]));
            }

            info.pe_list = pe_list;


            vector<int64_t> ifmap_offset;

            uint64_t ifmap_size = ifmap_height * ifmap_width * channels * word_size * batch_size * ifmap_source_list.size();
            uint64_t filter_size = 0;
            if (type == CONV)
                filter_size = filter_height * filter_width * channels * num_filer * word_size;
            uint64_t ofmap_size = calcinfo.ofmap_height * calcinfo.ofmap_width * num_filer * word_size * batch_size;

            uint64_t ifmap_demand_size = calcinfo.ofmap_height * calcinfo.ofmap_width * num_filer * filter_height * filter_width * channels * word_size * batch_size;
            uint64_t filter_demand_size = filter_height * filter_width * channels * num_filer * word_size * batch_size;

            if (topo_arrays.size() == 0) {
                filter_offset = initial_filter_offset;
                ofmap_offset = initial_ifmap_offset + ifmap_size;
            } else {
                filter_offset = topo_arrays[topo_arrays.size() - 1].filter_offset_end;
                ofmap_offset = topo_arrays[topo_arrays.size() - 1].ofmap_offset_end;
            }

            if (ifmap_source_list[0] == -1) {
                ifmap_offset.push_back(initial_ifmap_offset);
                initial_ifmap_offset += ifmap_size;
            } else {
                for (size_t i = 0; i < ifmap_source_list.size(); i++)
                {
                    ifmap_offset.push_back(topo_arrays[ifmap_source_list[i]].ofmap_offset);
                }
            }

            if (filter_source_list[0] != -1) {
                filter_offset = topo_arrays[filter_source_list[0]].ofmap_offset;
                filter_size = 0;
            }

            info.ifmap_offset = ifmap_offset;
            info.filter_offset = filter_offset;
            
            if (is_prefetch_demand) {
                info.filter_demand_offset = info.filter_offset + filter_size;
                info.filter_offset_end = info.filter_demand_offset + filter_demand_size;

                info.ofmap_offset = ofmap_offset + ifmap_demand_size;
                info.ofmap_offset_end = info.ofmap_offset + ofmap_size;
            } else {
                info.filter_demand_offset = info.filter_offset;
                info.filter_offset_end = info.filter_offset + filter_size;

                info.ofmap_offset = ofmap_offset;
                info.ofmap_offset_end = info.ofmap_offset + ofmap_size;
            }

            topo_arrays.push_back(info);
        }
    }

    

    num_layers = topo_arrays.size();
}

OffsetInfo Topology::get_layer_offsets(int64_t layer_id)
{
    OffsetInfo offsetInfo;
    offsetInfo.ifmap_offset = topo_arrays[layer_id].ifmap_offset;
    offsetInfo.filter_offset = topo_arrays[layer_id].filter_offset;
    offsetInfo.ofmap_offset = topo_arrays[layer_id].ofmap_offset;
    return offsetInfo;
}

int64_t Topology::get_layer_num_ofmap_px(int64_t layer_id)
{
    int64_t ofmap_px = (int64_t)1;
    ofmap_px *= calc_topo_arrays[layer_id].ofmap_height;
    ofmap_px *= calc_topo_arrays[layer_id].ofmap_width;
    ofmap_px *= topo_arrays[layer_id].num_filer;
    return ofmap_px;
}

#endif