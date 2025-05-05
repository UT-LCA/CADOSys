#ifndef _memory_map_h
#define _memory_map_h

#include <vector>

using namespace std;

class MemoryMap {
    public:
        MemoryMap();
        void set_single_bank_params(uint64_t filter_offset, uint64_t ofmap_offset);

        int64_t num_mappings;
        int64_t num_banks;

        vector<uint64_t> ifmap_map_list;
        vector<uint64_t> filter_map_list;
        vector<uint64_t> ofmap_map_list;

        bool map_data_valid = false;
};

MemoryMap::MemoryMap() {
    num_mappings = 1;
    num_banks = 1;
    map_data_valid = false;
}

void MemoryMap::set_single_bank_params(uint64_t filter_offset, uint64_t ofmap_offset) {
    num_mappings = 1;
    num_banks = 1;

    ifmap_map_list.push_back(filter_offset);
    filter_map_list.push_back(ofmap_offset);
    ofmap_map_list.push_back((uint64_t)-1);

    map_data_valid = true;
}

#endif