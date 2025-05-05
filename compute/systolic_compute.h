#ifndef _systolic_compute_h
#define _systolic_compute_h

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <algorithm>

#include "../scale_config.h"
#include "../topology_utils.h"

#include "../memory/double_buffer_scratchpad_mem.h"

class SystolicCompute {
public:
    SystolicCompute();
    // ~SystolicCompute() {}
    virtual void set_params(Config *config, xt::xarray<int64_t> &ifmap_op_mat, xt::xarray<int64_t> &filter_op_mat, xt::xarray<int64_t> &ofmap_op_mat, int num_pe) = 0;

    virtual xt::xarray<int64_t> get_ifmap_prefetch_matrices() = 0;
    virtual xt::xarray<int64_t> get_filter_prefetch_matrices() = 0;

    virtual vector<xt::xarray<int64_t>> get_ifmap_demand_matrices() = 0;
    virtual vector<xt::xarray<int64_t>> get_filter_demand_matrices() = 0;
    virtual vector<xt::xarray<int64_t>> get_ofmap_demand_matrices() = 0;

    virtual float get_avg_mapping_efficiency() = 0;
    virtual float get_avg_compute_utilization() = 0;
};

SystolicCompute::SystolicCompute() {

}

#endif