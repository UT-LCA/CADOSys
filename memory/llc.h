#ifndef _llc_h
#define _llc_h

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <vector>
#include <set>
#include <string>
#include <unordered_map>

#include "dram.h"

using namespace std;

typedef struct
{
    int64_t read_hit;
    int64_t read_miss_all;
    int64_t read_miss_conflict;
    int64_t write_hit;
    int64_t write_miss_all;
    int64_t write_miss_conflict;
} LLCStats;

enum class Replacement
{
    LRU,
    RRIP
};

class CacheContent
{
public:
    CacheContent(int64_t tag_bits);
    CacheContent(int64_t tag_bits, bool dirty_bit);
    int64_t tag_bits;
    int32_t rrip_bits = 3;
    bool dirty_bit;
};

CacheContent::CacheContent(int64_t tag_bits)
{
    this->tag_bits = tag_bits;
    this->dirty_bit = false;
}

CacheContent::CacheContent(int64_t tag_bits, bool dirty_bit)
{
    this->tag_bits = tag_bits;
    this->dirty_bit = dirty_bit;
}

class CacheSet
{
public:
    CacheSet(LLCStats *stats, Replacement replacement, int64_t set_associativity, string partition);
    bool service_read(int64_t tag_bits, int partition);
    bool service_write(int64_t tag_bits, int partition);

private:
    LLCStats *stats;
    Replacement replacement;
    int64_t set_associativity;

    int number_of_partitions;
    vector<vector<CacheContent *>> contents;
    vector<int> capacities;

    int is_read_hit(int64_t tag_bits, int partition);
    int is_write_hit(int64_t tag_bits, int partition);
    void update_queue_lru(int index, int partition);
    void replace_queue_lru(int64_t tag_bits, int partition);
    void update_queue_rrip(int index, int partition);
    void replace_queue_rrip(int64_t tag_bits, int partition);
};

CacheSet::CacheSet(LLCStats *stats, Replacement replacement, int64_t set_associativity, string partition)
{
    this->stats = stats;
    this->replacement = replacement;
    this->set_associativity = set_associativity;

    vector<string> eles_per_partititon;
    stringstream ss(partition);

    while (ss.good())
    {
        string substr;
        getline(ss, substr, ',');
        eles_per_partititon.push_back(substr);
    }

    number_of_partitions = eles_per_partititon.size();
    for (size_t i = 0; i < number_of_partitions; i++)
    {
        contents.push_back(vector<CacheContent *>());
        capacities.push_back(stoi(eles_per_partititon[i]));

        for (int j = 0; j < stoi(eles_per_partititon[i]); j++)
        {
            CacheContent *content = new CacheContent(-1);
            contents[i].push_back(content);
        }
    }
}

void CacheSet::update_queue_lru(int index, int partition)
{
    CacheContent *content = contents[partition][index];
    for (int i = index; i > 0; i--)
    {
        contents[partition][i] = contents[partition][i - 1];
    }
    contents[partition][0] = new CacheContent(content->tag_bits);
}

void CacheSet::replace_queue_lru(int64_t tag_bits, int partition)
{
    for (int i = capacities[partition] - 1; i > 0; i--)
    {
        contents[partition][i] = contents[partition][i - 1];
    }
    contents[partition][0] = new CacheContent(tag_bits);
}

// void CacheSet::replace_queue_lru(int64_t tag_bits, int partition)
// {
//     if (contents[partition].size() < capacities[partition])
//     {
//         CacheContent *content = new CacheContent(tag_bits);
//         contents[partition].push_back(content);
//     }
//     else
//     {
//         for (int i = capacities[partition] - 1; i > 0; i--)
//         {
//             contents[partition][i] = contents[partition][i - 1];
//         }
//         contents[partition][0] = new CacheContent(tag_bits);
//     }
// }

void CacheSet::update_queue_rrip(int index, int partition)
{
    return;
}

void CacheSet::replace_queue_rrip(int64_t tag_bits, int partition)
{
    while (1) {
        for (int i = 0; i < contents[partition].size(); i++) {
            if (contents[partition][i]->rrip_bits == 3) {
                contents[partition][i]->tag_bits = tag_bits;
                contents[partition][i]->rrip_bits = 2;
                return;
            }
        }
        for (int i = 0; i < contents[partition].size(); i++) {
            contents[partition][i]->rrip_bits++;
        }
    }
}

int CacheSet::is_read_hit(int64_t tag_bits, int partition)
{
    for (int i = 0; i < contents[partition].size(); i++)
    {
        if (contents[partition][i]->tag_bits == tag_bits) {
            contents[partition][i]->rrip_bits = 0;
            return i;
        }
    }
    if (contents[partition].size() == capacities[partition])
        stats->read_miss_conflict++;
    return -1;
}

int CacheSet::is_write_hit(int64_t tag_bits, int partition)
{
    for (int i = 0; i < contents[partition].size(); i++)
    {
        if (contents[partition][i]->tag_bits == tag_bits) {
            contents[partition][i]->rrip_bits = 0;
            return i;
        }
            
    }
    if (contents[partition].size() == capacities[partition])
        stats->write_miss_conflict++;
    return -1;
}

bool CacheSet::service_read(int64_t tag_bits, int partition)
{
    // return true;
    int index = is_read_hit(tag_bits, partition);
    if (index == -1)
    {
        if (replacement == Replacement::LRU)
            replace_queue_lru(tag_bits, partition);
        else if (replacement == Replacement::RRIP)
            replace_queue_rrip(tag_bits, partition);
        return false;
    }
    else
    {
        if (replacement == Replacement::LRU)
            update_queue_lru(index, partition);
        else if (replacement == Replacement::RRIP)
            update_queue_rrip(index, partition);
        return true;
    }
}

bool CacheSet::service_write(int64_t tag_bits, int partition)
{
    // return true;
    int index = is_write_hit(tag_bits, partition);
    if (index == -1)
    {
        if (replacement == Replacement::LRU)
            replace_queue_lru(tag_bits, partition);
        else if (replacement == Replacement::RRIP)
            replace_queue_rrip(tag_bits, partition);
        return false;
    }
    else
    {
        if (replacement == Replacement::LRU)
            update_queue_lru(index, partition);
        else if (replacement == Replacement::RRIP)
            update_queue_rrip(index, partition);
        return true;
    }
}

class LLC
{
public:
    LLC();
    void set_params(DRAM *dram, int64_t total_size_bytes, int64_t cache_line_size, int64_t hit_latency, int64_t set_associativity, string partition, bool is_always_hit, bool is_bypassing);
    int64_t get_latency() { return hit_latency; }
    int64_t service_read(set<int64_t> *incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset);
    int64_t service_write(set<int64_t> *incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset);
    int64_t service_read(xt::xarray<int64_t> incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset);
    int64_t service_write(xt::xarray<int64_t> incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset);
    void dump_stats();
    LLCStats get_llc_stats() { return stats; }
    void inc_read_miss_conflict() { stats.read_miss_conflict++; }
    void inc_write_miss_conflict() { stats.write_miss_conflict++; }

private:
    DRAM *dram;
    vector<CacheSet *> cacheSets;
    int64_t total_size_bytes;
    int64_t cache_line_size;
    int64_t hit_latency;
    int64_t miss_latency;
    int number_of_partitions;
    bool is_always_hit;
    bool is_bypassing;

    LLCStats stats;

    Replacement replacement;
    int64_t set_associativity;

    int number_of_sets;
    int set_bits;
    int offset_bits;

    int get_set_index(int64_t addr);
    int64_t get_tag(int64_t addr);

    int num_mshr;

    unordered_map<int64_t, int64_t> mshr;

    int64_t last_addr_no_offset = -1;
};

LLC::LLC()
{
    total_size_bytes = 1 * 1024 * 1024;
    cache_line_size = 64;
    hit_latency = 1;
    set_associativity = 4;
    number_of_partitions = 1;
    is_always_hit = false;
    num_mshr = 8;

    last_addr_no_offset = -1;

    stats.read_hit = 0;
    stats.read_miss_conflict = 0;
    stats.read_miss_all = 0;

    stats.write_hit = 0;
    stats.write_miss_conflict = 0;
    stats.write_miss_all = 0;
}

void LLC::set_params(DRAM *dram, int64_t total_size_bytes, int64_t cache_line_size, int64_t hit_latency, int64_t set_associativity, string partition, bool is_always_hit, bool is_bypassing)
{
    this->dram = dram;
    this->total_size_bytes = total_size_bytes;
    this->cache_line_size = cache_line_size;
    this->hit_latency = hit_latency;
    this->set_associativity = set_associativity;
    this->is_always_hit = is_always_hit;
    this->miss_latency = dram->get_latency();
    this->is_bypassing = is_bypassing;

    // this->hit_latency = 2;
    // this->is_bypassing = false;

    replacement = Replacement::RRIP;

    number_of_sets = (int)(total_size_bytes / (cache_line_size * (int64_t)pow(2, set_associativity)));
    set_bits = int(log2(number_of_sets));
    offset_bits = int(log2(cache_line_size));

    cout << "number_of_sets is " << number_of_sets << endl;

    for (int i = 0; i < number_of_sets; i++)
    {
        CacheSet *cacheSet = new CacheSet(&stats, replacement, set_associativity, partition);
        cacheSets.push_back(cacheSet);
    }
}

int64_t LLC::service_read(set<int64_t> *incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset)
{
    int64_t out_cycle = incoming_cycles_arr;
    int64_t offset = 0;

    int64_t addr_no_offset = -1;
    if (reset)
        last_addr_no_offset = -1;

    if (is_bypassing && reset) return (out_cycle + hit_latency);
    if (is_bypassing && !reset) return (out_cycle);

    for (auto i = incoming_requests->begin(); i != incoming_requests->end(); ++i)
    {
        int64_t addr = (*i);
        if (addr == -1)
            continue;

        addr_no_offset = addr >> offset_bits;

        if (addr_no_offset == last_addr_no_offset) continue;

        bool is_hit = false;
        if (is_always_hit) {
            is_hit = true;
        } else {
            int cache_set_id = get_set_index(addr);
            int64_t tag_bits = get_tag(addr);
            is_hit = cacheSets[cache_set_id]->service_read(tag_bits, partition);
        }

        if (is_hit)
        {
            offset += hit_latency;
            stats.read_hit++;
        }
        else
        {
            offset += miss_latency;
            stats.read_miss_all++;
        }
        last_addr_no_offset = addr_no_offset;
    }
    out_cycle += offset;
    return out_cycle;
}

int64_t LLC::service_write(set<int64_t> *incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset)
{
    int64_t out_cycle = incoming_cycles_arr;
    int64_t offset = 0;

    int64_t addr_no_offset = -1;
    if (reset)
        last_addr_no_offset = -1;

    if (is_bypassing && reset) return (out_cycle + hit_latency);
    if (is_bypassing && !reset) return (out_cycle);

    for (auto i = incoming_requests->begin(); i != incoming_requests->end(); ++i)
    {
        int64_t addr = (*i);
        if (addr == -1)
            continue;

        addr_no_offset = addr >> offset_bits;

        if (addr_no_offset == last_addr_no_offset) continue;

        bool is_hit = false;
        if (is_always_hit) {
            is_hit = true;
        } else {
            int cache_set_id = get_set_index(addr);
            int64_t tag_bits = get_tag(addr);
            is_hit = cacheSets[cache_set_id]->service_write(tag_bits, partition);
        }

        if (is_hit)
        {
            offset += hit_latency;
            stats.write_hit++;
        }
        else
        {
            offset += miss_latency;
            stats.write_miss_all++;
        }
        last_addr_no_offset = addr_no_offset;
    }
    out_cycle += offset;
    return out_cycle;
}

int64_t LLC::service_read(xt::xarray<int64_t> incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset)
{
    int64_t out_cycle = incoming_cycles_arr;
    int64_t offset = 0;
    
    int64_t addr_no_offset = -1;
    if (reset)
        last_addr_no_offset = -1;

    if (is_bypassing && reset) return (out_cycle + hit_latency);
    if (is_bypassing && !reset) return (out_cycle);

    for (int64_t addr : incoming_requests)
    {
        if (addr == -1)
            continue;

        addr_no_offset = addr >> offset_bits;

        if (addr_no_offset == last_addr_no_offset) continue;

        bool is_hit = false;
        if (is_always_hit) {
            is_hit = true;
        } else {
            int cache_set_id = get_set_index(addr);
            int64_t tag_bits = get_tag(addr);
            is_hit = cacheSets[cache_set_id]->service_read(tag_bits, partition);
        }

        if (is_hit)
        {
            offset += hit_latency;
            stats.read_hit++;
        }
        else
        {
            offset += miss_latency;
            stats.read_miss_all++;
        }
        last_addr_no_offset = addr_no_offset;
    }
    out_cycle += offset;
    return out_cycle;
}

int64_t LLC::service_write(xt::xarray<int64_t> incoming_requests, int64_t incoming_cycles_arr, int partition, bool reset)
{
    int64_t out_cycle = incoming_cycles_arr;
    int64_t offset = 0;

    int64_t addr_no_offset = -1;
    if (reset)
        last_addr_no_offset = -1;

    if (is_bypassing && reset) return (out_cycle + hit_latency);
    if (is_bypassing && !reset) return (out_cycle);

    for (int64_t addr : incoming_requests)
    {
        if (addr == -1)
            continue;

        addr_no_offset = addr >> offset_bits;

        if (addr_no_offset == last_addr_no_offset) continue;

        bool is_hit = false;
        if (is_always_hit) {
            is_hit = true;
        } else {
            int cache_set_id = get_set_index(addr);
            int64_t tag_bits = get_tag(addr);
            is_hit = cacheSets[cache_set_id]->service_write(tag_bits, partition);
        }

        if (is_hit)
        {
            offset += hit_latency;
            stats.write_hit++;
        }
        else
        {
            offset += miss_latency;
            stats.write_miss_all++;
        }
        last_addr_no_offset = addr_no_offset;
    }
    out_cycle += offset;
    return out_cycle;
}

int LLC::get_set_index(int64_t addr)
{
    int64_t set_index = addr >> offset_bits;
    int64_t index_bits = (int64_t)(pow(2, set_bits)) - 1;
    set_index = set_index & index_bits;
    return (int)set_index;
}

int64_t LLC::get_tag(int64_t addr)
{
    int tag = (int)(addr >> (set_bits + offset_bits));
    return tag;
}

void LLC::dump_stats()
{
    cout << "llc.read_hit is " << stats.read_hit << endl;
    cout << "llc.read_miss_conflict is " << stats.read_miss_conflict << endl;
    cout << "llc.read_miss_all is " << stats.read_miss_all << endl;

    cout << "llc.write_hit is " << stats.write_hit << endl;
    cout << "llc.write_miss_conflict is " << stats.write_miss_conflict << endl;
    cout << "llc.write_miss_all is " << stats.write_miss_all << endl;
}

#endif