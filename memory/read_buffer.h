#ifndef _read_buffer_h
#define _read_buffer_h

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <vector>
#include <set>

using namespace std;

#include "llc.h"

class ReadBuffer {
public:
    ReadBuffer(bool verbose);
    void set_params(LLC* llc, int64_t total_size_bytes, int64_t word_size, float active_buf_frac, int64_t req_gen_bandwidth);
    void set_fetch_matrix(xt::xarray<int64_t> fetch_matrix_np);
    xt::xarray<int64_t> service_reads(xt::xarray<int64_t> incoming_requests_arr_np, xt::xarray<int64_t> incoming_cycles_arr, int llc_partition, bool trans);
    int64_t service_read(xt::xarray<int64_t> incoming_requests_arr_np, int64_t incoming_cycle, int llc_partition, bool trans);
    int64_t service_read(int request_line_id, int64_t incoming_cycle, int llc_partition, bool trans);
    int64_t get_hit_latency() { return hit_latency; }

    int64_t get_last_prefetch_cycle() { return last_prefetch_cycle; }
    void add_last_prefetch_cycle(int64_t cycle) { last_prefetch_cycle += cycle;}
private:
    LLC *llc;
    int64_t total_size_bytes;
    int64_t word_size; 
    float active_buf_frac;
    int64_t req_gen_bandwidth;

    xt::xarray<int64_t> fetch_matrix;
    int64_t last_prefetch_cycle;
    int64_t next_line_prefetch_idx;
    int64_t next_col_prefetch_idx;

    int64_t hit_latency;
    int64_t num_lines;
    int64_t num_active_buf_lines;
    int64_t num_prefetch_buf_lines;
    pair<int64_t, int64_t> active_buffer_set_limits;
    pair<int64_t, int64_t> prefetch_buffer_set_limits;

    int64_t max_num_active_buf_lines;
    int64_t max_num_prefetch_buf_lines;

    int64_t total_size_elems;
    int64_t active_buf_size;
    int64_t prefetch_buf_size;

    int64_t num_access;
    int64_t elems_per_set;

    bool trans;

    bool verbose = false;
    bool active_buf_full_flag = false;
    bool hashed_buffer_valid = false;
    bool trace_valid = false;
    bool finished = false;

    vector<set<int64_t>*> hashed_buffer;
    vector<int64_t> hashed_line_id;
    vector<bool> hashed_has_content;

    void prepare_hashed_buffer();
    void prefetch_active_buffer(int64_t start_cycle, int llc_partition);
    int64_t active_buffer_hit(int64_t addr);
    void new_prefetch(int llc_partition);
};

ReadBuffer::ReadBuffer(bool verbose) {
    total_size_bytes = 2048;
    word_size = 1;
    active_buf_frac = 0.5;
    hit_latency = 1;

    num_lines = 0;
    num_active_buf_lines = 1;
    num_prefetch_buf_lines = 1;
    num_access = 0;

    last_prefetch_cycle = -1;
    trans = false;

    req_gen_bandwidth = 32;
    elems_per_set = req_gen_bandwidth;

    total_size_elems = total_size_bytes / word_size;
    active_buf_size = (int64_t) (total_size_elems * active_buf_frac);
    prefetch_buf_size = total_size_elems - active_buf_size;

    max_num_active_buf_lines = (active_buf_size + elems_per_set - 1) / elems_per_set;
    max_num_prefetch_buf_lines = (prefetch_buf_size + elems_per_set - 1) / elems_per_set;

    this->verbose = verbose;
}

void ReadBuffer::set_params(LLC* llc, int64_t total_size_bytes, int64_t word_size, float active_buf_frac, int64_t req_gen_bandwidth) {
    this->llc = llc;
    this->word_size = word_size;
    this->active_buf_frac = active_buf_frac;
    this->req_gen_bandwidth = req_gen_bandwidth;

    total_size_elems = total_size_bytes / word_size;
    active_buf_size = (int64_t) (total_size_elems * active_buf_frac);
    prefetch_buf_size = total_size_elems - active_buf_size;

    max_num_active_buf_lines = (active_buf_size + elems_per_set - 1) / elems_per_set;
    max_num_prefetch_buf_lines = (prefetch_buf_size + elems_per_set - 1) / elems_per_set;
}

void ReadBuffer::set_fetch_matrix(xt::xarray<int64_t> fetch_matrix_np) {
    cout << "ReadBuffer::set_fetch_matrix" << endl;
    int64_t src_rows = fetch_matrix_np.shape()[0];
    int64_t src_cols = fetch_matrix_np.shape()[1];
    int64_t num_elems = src_rows * src_cols;
    int64_t num_lines = (num_elems + req_gen_bandwidth - 1) / req_gen_bandwidth;
    cout << src_rows << ", " << src_cols << endl;
    cout << num_lines << ", " << req_gen_bandwidth << endl;
    fetch_matrix = xt::ones<int64_t>({num_lines, req_gen_bandwidth}) * -1;

    for (int64_t i = 0; i < num_elems; i++) {
        int64_t src_row = i / fetch_matrix_np.shape()[1];
        int64_t src_col = i % fetch_matrix_np.shape()[1];

        int64_t dest_row = i / req_gen_bandwidth;
        int64_t dest_col = i % req_gen_bandwidth;

        fetch_matrix(dest_row, dest_col) = fetch_matrix_np(src_row, src_col);
    }

    last_prefetch_cycle = -1;
    prepare_hashed_buffer();
    // cout << fetch_matrix << endl;
}

void ReadBuffer::prepare_hashed_buffer() {
    cout << "prepare_hashed_buffer" << endl;
    // int64_t elems_per_set = (total_size_elems + 99) / 100;
    elems_per_set = req_gen_bandwidth;

    int64_t prefetch_rows = fetch_matrix.shape()[0];
    int64_t prefetch_cols = fetch_matrix.shape()[1];

    int64_t line_id = 0;

    hashed_buffer.clear();
    hashed_line_id.clear();
    hashed_has_content.clear();

    active_buf_full_flag = false;
    finished = false;
        
    for (int64_t r = 0; r < prefetch_rows; r++) {
        // set<int64_t> *this_set = new set<int64_t>(elems_per_set);
        set<int64_t> *this_set = new set<int64_t>();
        bool has_content = false;
        for (int64_t c = 0; c < prefetch_cols; c++) {
            int64_t elem = fetch_matrix(r, c);
            if (elem != -1)
                has_content = true;
            this_set->insert(elem);
        }
        hashed_line_id.push_back(line_id);
        hashed_has_content.push_back(has_content);
        if (has_content) {
            hashed_buffer.push_back(this_set);
            line_id++;

            // for (int64_t c = 0; c < prefetch_cols; c++) {
            //     int64_t elem = fetch_matrix(r, c);
            //     std::cout << elem << ", ";
            // }
            // cout << endl;
        }
            
    }

    max_num_active_buf_lines = (active_buf_size + elems_per_set - 1) / elems_per_set;
    max_num_prefetch_buf_lines = (prefetch_buf_size + elems_per_set - 1) / elems_per_set;
    int64_t num_lines = line_id;

    cout << "num_lines is " << num_lines << endl;
        

    if (num_lines > max_num_active_buf_lines)
        num_active_buf_lines = max_num_active_buf_lines;
    else
        num_active_buf_lines = num_lines;

    int64_t remaining_lines = num_lines - num_active_buf_lines;

    if (remaining_lines > max_num_prefetch_buf_lines)
        num_prefetch_buf_lines = max_num_prefetch_buf_lines;
    else
        num_prefetch_buf_lines = remaining_lines;

    this->num_lines = num_lines;
    hashed_buffer_valid = true;
}

int64_t ReadBuffer::active_buffer_hit(int64_t addr) {
    int64_t start_id = active_buffer_set_limits.first;
    int64_t end_id = active_buffer_set_limits.second;

    int64_t hit_id = -1;

    if (start_id < end_id) {
        for (int line_id = start_id; line_id < end_id; line_id++) {
            set<int64_t> *this_set = hashed_buffer[line_id];
            if (this_set->find(addr) != this_set->end())                       
                return true;
        }        
    } else {
        for (int line_id = start_id; line_id < num_lines; line_id++) {
            set<int64_t> *this_set = hashed_buffer[line_id];
            if (this_set->find(addr) != this_set->end())                       
                return true;
        } 

        for (int line_id = 0; line_id < end_id; line_id++) {
            set<int64_t> *this_set = hashed_buffer[line_id];
            if (this_set->find(addr) != this_set->end())                       
                return true;
        } 
    }

    return false;  
}


xt::xarray<int64_t> ReadBuffer::service_reads(xt::xarray<int64_t> incoming_requests_arr_np, xt::xarray<int64_t> incoming_cycles_arr, int llc_partition, bool trans) {
    this->trans = trans;
    if (!active_buf_full_flag) {
        int64_t start_cycle = incoming_cycles_arr(0);
        prefetch_active_buffer(start_cycle, llc_partition); 
    }

    xt::xarray<int64_t> out_cycles_arr = xt::zeros<int64_t>({incoming_requests_arr_np.shape()[0]});
    int64_t offset = hit_latency;

    for (int64_t i = 0; i < incoming_requests_arr_np.shape()[0]; i++) {
        int64_t cycle = max(incoming_cycles_arr(i), last_prefetch_cycle);
        xt::xarray<int64_t> request_line = xt::row(incoming_requests_arr_np, i);
        for (int64_t addr : request_line) {
            if (addr == -1)
                continue;

            while (!active_buffer_hit(addr)) {
                last_prefetch_cycle = max(last_prefetch_cycle, cycle);
                new_prefetch(llc_partition);
            }      
        }
        out_cycles_arr(i) = incoming_cycles_arr(i) + offset;
    }
    return out_cycles_arr;
}


int64_t ReadBuffer::service_read(xt::xarray<int64_t> request_line, int64_t incoming_cycle, int llc_partition, bool trans) {
    this->trans = trans;
    // cout << "service_read" << endl;
    if (!active_buf_full_flag) {
        int64_t start_cycle = incoming_cycle;
        prefetch_active_buffer(start_cycle, llc_partition); 
    }

    int64_t offset = hit_latency;
    int64_t cycle = max(incoming_cycle, last_prefetch_cycle);
    
    for (int64_t addr : request_line) {
        if (addr == -1)
            continue;

        while (!active_buffer_hit(addr)) {
            last_prefetch_cycle = max(last_prefetch_cycle, cycle);
            new_prefetch(llc_partition);
        }      
    }
    int64_t out_cycle = cycle + offset;
    return out_cycle;
}


int64_t ReadBuffer::service_read(int request_line_id, int64_t incoming_cycle, int llc_partition, bool trans) {
    this->trans = trans;
    int64_t line_id = hashed_line_id[request_line_id];

    // cout << "line_id is " << line_id << endl;
    // cout << "hashed_has_content[request_line_id] is " << hashed_has_content[request_line_id] << endl;

    int64_t offset = hit_latency;
    int64_t cycle = incoming_cycle;
    
    if (hashed_has_content[request_line_id]) {
        if (!active_buf_full_flag) {
            int64_t start_cycle = incoming_cycle;
            prefetch_active_buffer(start_cycle, llc_partition); 
        }

        if (line_id != 0 && line_id % max_num_active_buf_lines == 0 && !finished) {
            if (!finished) {
                cycle = max(incoming_cycle, last_prefetch_cycle);
                last_prefetch_cycle = max(last_prefetch_cycle, cycle);
                new_prefetch(llc_partition);
            }
            if (line_id == (num_lines - 1)) finished = true;
        }

        if (line_id == (num_lines - 1)) {
            if (!finished) {
                cycle = max(incoming_cycle, last_prefetch_cycle);
                last_prefetch_cycle = max(last_prefetch_cycle, cycle);
                new_prefetch(llc_partition);
            }
            finished = true;
        }
    }
        
    int64_t out_cycle = cycle + offset;
    return out_cycle;
}


void ReadBuffer::prefetch_active_buffer(int64_t start_cycle, int llc_partition) {
    int64_t fetch_lines = (active_buf_size + req_gen_bandwidth - 1) / req_gen_bandwidth;
    
    if (fetch_lines >= hashed_buffer.size()) {
        fetch_lines = hashed_buffer.size();
    }
        
    int64_t requested_data_size = fetch_lines * elems_per_set;
    num_access += requested_data_size;

    int64_t start_idx = 0;
    int64_t end_idx = fetch_lines;

    if (!trans) {
        for (int line_id = start_idx; line_id < end_idx; line_id++) {
            set<int64_t> *this_set = hashed_buffer[line_id];
            last_prefetch_cycle = llc->service_read(this_set, last_prefetch_cycle, llc_partition, (line_id + 1) % 2);
        } 
    } else {
        xt::xarray<int64_t> trans_hashed_buffer = xt::zeros<int64_t>({req_gen_bandwidth, fetch_lines});
        int row = 0;
        for (int line_id = start_idx; line_id < end_idx; line_id++) {
            set<int64_t> *this_set = hashed_buffer[line_id];
            int col = 0;
            for (auto i = this_set->begin(); i != this_set->end(); ++i) {
                int64_t addr = (*i);
                trans_hashed_buffer(col, row) = addr;
                col++;
            }
            row++;
        }

        for (int i = 0; i < req_gen_bandwidth; i++) {
            xt::xarray<int64_t> trans_line = xt::row(trans_hashed_buffer, i);
            last_prefetch_cycle = llc->service_read(trans_line, last_prefetch_cycle, llc_partition, (i + 1) % 2);
        }
    }

    trace_valid = true;

    int64_t active_buf_start_line_id = 0;
    int64_t active_buf_end_line_id = num_active_buf_lines;
    active_buffer_set_limits = {active_buf_start_line_id, active_buf_end_line_id};

    int64_t prefetch_buf_start_line_id = active_buf_end_line_id;
    int64_t prefetch_buf_end_line_id = prefetch_buf_start_line_id + num_prefetch_buf_lines;
    prefetch_buffer_set_limits = {prefetch_buf_start_line_id, prefetch_buf_end_line_id};

    active_buf_full_flag = true;
}

void ReadBuffer::new_prefetch(int llc_partition) {
    // cout << "new_prefetch at last_prefetch_cycle " << last_prefetch_cycle << endl;
    int64_t active_start = active_buffer_set_limits.first;
    int64_t active_end = active_buffer_set_limits.second;

    active_start = (active_start + num_prefetch_buf_lines) % num_lines;
    active_end = (active_start + num_active_buf_lines) % num_lines;
    int64_t prefetch_start = active_end;
    int64_t prefetch_end = (prefetch_start + num_prefetch_buf_lines) % num_lines;
        
    active_buffer_set_limits = {active_start, active_end};
    prefetch_buffer_set_limits = {prefetch_start, prefetch_end};
    
    int64_t start_idx = prefetch_start;
    int64_t fetch_lines = (prefetch_buf_size + elems_per_set - 1) / elems_per_set;
    int64_t end_idx = (start_idx + fetch_lines) % num_lines;
    int64_t requested_data_size = fetch_lines * elems_per_set;
    num_access += requested_data_size;

    // cout << "start_idx is " << start_idx << ", end_idx is " << end_idx << endl;

    if (!trans) {
        if (end_idx > start_idx) {
            for (int line_id = start_idx; line_id < end_idx; line_id++) {
                set<int64_t> *this_set = hashed_buffer[line_id];
                last_prefetch_cycle = llc->service_read(this_set, last_prefetch_cycle, llc_partition, (line_id + 1) % 2);
            }        
        } else {
            cout << "read_buffer end_idx < start_idx" << endl;
            cout << "start_idx is " << start_idx << ", end_idx is " << end_idx << ", num_lines is " << num_lines << endl;
            for (int line_id = start_idx; line_id < num_lines; line_id++) {
                set<int64_t> *this_set = hashed_buffer[line_id];
                last_prefetch_cycle = llc->service_read(this_set, last_prefetch_cycle, llc_partition, (line_id + 1) % 2);
            } 

            for (int line_id = 0; line_id < end_idx; line_id++) {
                set<int64_t> *this_set = hashed_buffer[line_id];
                last_prefetch_cycle = llc->service_read(this_set, last_prefetch_cycle, llc_partition, (line_id + 1) % 2);
            } 
        }
    } else {

        xt::xarray<int64_t> trans_hashed_buffer = xt::zeros<int64_t>({req_gen_bandwidth, fetch_lines});
        int row = 0;
        if (end_idx > start_idx) {
            for (int line_id = start_idx; line_id < end_idx; line_id++) {
                set<int64_t> *this_set = hashed_buffer[line_id];
                int col = 0;
                for (auto i = this_set->begin(); i != this_set->end(); ++i) {
                    int64_t addr = (*i);
                    trans_hashed_buffer(col, row) = addr;
                    col++;
                }
                row++;
            }
        } else {
            cout << "read_buffer end_idx < start_idx" << endl;
            cout << "start_idx is " << start_idx << ", end_idx is " << end_idx << ", num_lines is " << num_lines << endl;
            for (int line_id = start_idx; line_id < num_lines; line_id++) {
                set<int64_t> *this_set = hashed_buffer[line_id];
                int col = 0;
                for (auto i = this_set->begin(); i != this_set->end(); ++i) {
                    int64_t addr = (*i);
                    trans_hashed_buffer(col, row) = addr;
                    col++;
                }
                row++;
            }

            for (int line_id = 0; line_id < end_idx; line_id++) {
                set<int64_t> *this_set = hashed_buffer[line_id];
                int col = 0;
                for (auto i = this_set->begin(); i != this_set->end(); ++i) {
                    int64_t addr = (*i);
                    trans_hashed_buffer(col, row) = addr;
                    col++;
                }
                row++;
            }
        }

        for (int i = 0; i < req_gen_bandwidth; i++) {
            xt::xarray<int64_t> trans_line = xt::row(trans_hashed_buffer, i);
            last_prefetch_cycle = llc->service_read(trans_line, last_prefetch_cycle, llc_partition, (i + 1) % 2);
        }

    }
    // cout << "finish new_prefetch at last_prefetch_cycle " << last_prefetch_cycle << endl;
}

#endif