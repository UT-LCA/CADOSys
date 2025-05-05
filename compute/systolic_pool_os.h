#ifndef _systolic_pool_os_h
#define _systolic_pool_os_h

#include "systolic_compute.h"

#include <chrono>
#include <iostream>

using namespace std::chrono;

class SystolicPoolOs : public SystolicCompute
{
public:
    SystolicPoolOs();
    ~SystolicPoolOs() {}
    void set_params(Config *config, xt::xarray<int64_t> &ifmap_op_mat, xt::xarray<int64_t> &filter_op_mat, xt::xarray<int64_t> &ofmap_op_mat, int num_pe);

    xt::xarray<int64_t> get_ifmap_prefetch_matrices();
    xt::xarray<int64_t> get_filter_prefetch_matrices();

    vector<xt::xarray<int64_t>> get_ifmap_demand_matrices();
    vector<xt::xarray<int64_t>> get_filter_demand_matrices();
    vector<xt::xarray<int64_t>> get_ofmap_demand_matrices();

    float get_avg_mapping_efficiency();
    float get_avg_compute_utilization();

private:
    xt::xarray<int64_t> skew_matrix(xt::xarray<int64_t> input_matrix_np);
    Config *config;
    xt::xarray<int64_t> ifmap_op_mat;
    xt::xarray<int64_t> filter_op_mat;
    xt::xarray<int64_t> ofmap_op_mat;

    xt::xarray<int64_t> ifmap_op_mat_trans;

    xt::xarray<int64_t> ifmap_prefetch_matrix;
    xt::xarray<int64_t> filter_prefetch_matrix;

    xt::xarray<int64_t> ifmap_demand_matrix;
    xt::xarray<int64_t> filter_demand_matrix;
    xt::xarray<int64_t> ofmap_demand_matrix;

    int num_pe;

    int ifmap_col;
    int filter_row;

    int Sr;
    int Sc;
    int T;
    int W;

    int arr_row;
    int arr_col;

    int row_fold;
    int col_fold;

    int64_t ifmap_reads;
    int64_t filter_reads;
    int64_t ofmap_writes;

    vector<float> mapping_efficiency_per_fold;
    vector<float> compute_utility_per_fold;
};

SystolicPoolOs::SystolicPoolOs()
{
    ifmap_reads = 0;
    filter_reads = 0;
    ofmap_writes = 0;
}

void SystolicPoolOs::set_params(Config *config, xt::xarray<int64_t> &ifmap_op_mat, xt::xarray<int64_t> &filter_op_mat, xt::xarray<int64_t> &ofmap_op_mat, int num_pe)
{
    this->config = config;
    this->ifmap_op_mat = ifmap_op_mat;
    this->filter_op_mat = filter_op_mat;
    this->ofmap_op_mat = ofmap_op_mat;
    this->num_pe = num_pe;

    ifmap_col = this->ifmap_op_mat.shape()[1];
    filter_row = this->filter_op_mat.shape()[0];

    ifmap_op_mat_trans = xt::transpose(ifmap_op_mat);

    Sr = ifmap_op_mat.shape()[0];
    Sc = filter_op_mat.shape()[1];
    T = ifmap_op_mat.shape()[1];

    // Sr = ifmap_op_mat.shape()[1];
    // Sc = filter_op_mat.shape()[1];
    // T = ifmap_op_mat.shape()[0];
    // W = Sr / Sc;
    // if (W <= 1) W = 1;

    auto arrayDims = config->get_array_dims();
    arr_row = arrayDims.arr_h;
    arr_col = arrayDims.arr_w;

    row_fold = (Sr + arr_row - 1) / (arr_row * num_pe);
    col_fold = (Sc + arr_col - 1) / arr_col;

    cout << "Sr " << Sr << ", Sc " << Sc  << ", T " << T << endl;
    cout << "row_fold " << row_fold << ", col_fold " << col_fold << endl;
}

xt::xarray<int64_t> SystolicPoolOs::get_ifmap_prefetch_matrices()
{
    int basic_iter = ifmap_op_mat.shape()[0];
    ifmap_prefetch_matrix = xt::ones<int64_t>({basic_iter * row_fold, arr_row});
    // ifmap_demand_matrix = xt::ones<int64_t>({Sr, arr_row});

    cout << "get_ifmap_prefetch_matrices()" << endl;
    cout << "ifmap_op_mat shape is " << ifmap_op_mat.shape()[0] << ", " << ifmap_op_mat.shape()[1] << endl;
    cout << "ifmap_op_mat_trans shape is " << ifmap_op_mat_trans.shape()[0] << ", " << ifmap_op_mat_trans.shape()[1] << endl;
    cout << "ofmap_op_mat shape is " << ofmap_op_mat.shape()[0] << ", " << ofmap_op_mat.shape()[1] << endl;

    for (int fr = 0; fr < row_fold; fr++)
    {
        // cout << "fr " << fr << " of " << row_fold << endl;
        int start_col_idx = fr * arr_row;
        int end_col_idx = min(start_col_idx + arr_row, Sr);
        // cout << "start_row_idx " << start_row_idx << ", end_row_idx " << end_row_idx << endl;

        int delta = arr_row - (end_col_idx - start_col_idx);

        xt::xarray<int64_t> this_fold_prefetch = xt::view(ifmap_op_mat, xt::all(), xt::range(start_col_idx, end_col_idx));

        if (delta > 0)
        {
            xt::xarray<int64_t> null_req_mat = xt::ones<int64_t>({T, delta}) * -1;
            this_fold_prefetch = xt::concatenate(xtuple(this_fold_prefetch, null_req_mat), 1);
        }

        xt::view(ifmap_prefetch_matrix, xt::range(basic_iter * fr, basic_iter * (fr + 1)), xt::all()) = this_fold_prefetch;
    }

    int M = ifmap_prefetch_matrix.shape()[0];
    int N = ifmap_prefetch_matrix.shape()[1];

    cout << "ifmap_prefetch_matrix shape is " << M << ", " << N << endl;

    int num_elems = M * N;
    int num_diags = M + N;
    xt::xarray<int64_t> prefetches = xt::zeros<int64_t>({1, num_elems});
    int idx = 0;

    for (int diag_id = 0; diag_id < num_diags; diag_id++)
    {
        int max_row_id = min(diag_id, M - 1);
        int min_row_id = max(0, diag_id - N + 1);
        int valid_rows = max_row_id - min_row_id + 1;

        for (int offset = 0; offset < valid_rows; offset++)
        {
            int row_id = max_row_id - offset;
            int col_id = diag_id - row_id;

            int64_t elem = ifmap_prefetch_matrix(row_id, col_id);
            prefetches(0, idx) = elem;
            idx++;
        }
    }

    ifmap_prefetch_matrix = prefetches;
    return ifmap_prefetch_matrix;
}

xt::xarray<int64_t> SystolicPoolOs::get_filter_prefetch_matrices()
{
    cout << "get_filter_prefetch_matrices()" << endl;
    cout << "filter_op_mat shape is " << filter_op_mat.shape()[0] << ", " << filter_op_mat.shape()[1] << endl;
    int basic_iter = filter_op_mat.shape()[0];
    filter_prefetch_matrix = xt::ones<int64_t>({basic_iter * col_fold, arr_row});

    for (int fc = 0; fc < col_fold; fc++)
    {
        // cout << "fc " << fc << " of " << col_fold << endl;
        int col_start_id = fc * arr_col;
        int col_end_id = min(col_start_id + arr_col, Sc);

        int delta = arr_col - (col_end_id - col_start_id);

        xt::xarray<int64_t> this_fold_prefetch = xt::view(filter_op_mat, xt::all(), xt::range(col_start_id, col_end_id));

        if (delta > 0)
        {
            xt::xarray<int64_t> null_req_mat = xt::ones<int64_t>({Sr, delta}) * -1;
            this_fold_prefetch = xt::concatenate(xtuple(this_fold_prefetch, null_req_mat), 1);
        }

        xt::view(filter_prefetch_matrix, xt::range(basic_iter * fc, basic_iter * (fc + 1)), xt::all()) = this_fold_prefetch;
    }

    int M = filter_prefetch_matrix.shape()[0];
    int N = filter_prefetch_matrix.shape()[1];

    cout << "filter_prefetch_matrix shape is " << M << ", " << N << endl;

    return filter_prefetch_matrix;
}

vector<xt::xarray<int64_t>> SystolicPoolOs::get_ifmap_demand_matrices()
{
    cout << "get_ifmap_demand_matrices()" << endl;

    vector<xt::xarray<int64_t>> ifmap_demand_matrix_vector;

    for (int i = 0; i < num_pe; i++) {
        int inter_fold_gap_suffix = arr_col - 1;
        xt::xarray<int64_t> inter_fold_gap_suffix_mat = xt::ones<int64_t>({inter_fold_gap_suffix, arr_row}) * -1;

        // int basic_iter = arr_row - 1 + arr_col - 1 + ifmap_op_mat.shape()[1];
        int basic_iter = arr_col - 1 + ifmap_op_mat.shape()[1];
        ifmap_demand_matrix = xt::ones<int64_t>({basic_iter * col_fold * row_fold, arr_row});

        for (int fc = 0; fc < col_fold; fc++)
        {
            for (int fr = 0; fr < row_fold; fr++)
            {
                int index = fc * row_fold * num_pe + i * row_fold + fr;

                int row_start_id = (i * row_fold + fr) * arr_row;
                int row_end_idx = min(row_start_id + arr_row, Sr);
                int delta = arr_row - (row_end_idx - row_start_id);

                xt::xarray<int64_t> this_fold_demand = xt::view(ifmap_op_mat_trans, xt::all(), xt::range(row_start_id, row_end_idx));
                ifmap_reads += this_fold_demand.shape()[0] * this_fold_demand.shape()[1];

                if (delta > 0)
                {
                    xt::xarray<int64_t> null_req_mat = xt::ones<int64_t>({T, delta}) * -1;
                    this_fold_demand = xt::concatenate(xtuple(this_fold_demand, null_req_mat), 1);
                }
        
                this_fold_demand = xt::concatenate(xtuple(this_fold_demand, inter_fold_gap_suffix_mat), 0);
                xt::view(ifmap_demand_matrix, xt::range(basic_iter * index, basic_iter * (index + 1)), xt::all()) = this_fold_demand;
            }
        }
        cout << "ifmap_demand_matrix shape is " << ifmap_demand_matrix.shape()[0] << ", " << ifmap_demand_matrix.shape()[1] << endl;
        cout << ifmap_demand_matrix << endl;
        ifmap_demand_matrix_vector.push_back(ifmap_demand_matrix);
    }
    return ifmap_demand_matrix_vector;
}

vector<xt::xarray<int64_t>> SystolicPoolOs::get_filter_demand_matrices()
{
    cout << "get_filter_demand_matrices()" << endl;

    vector<xt::xarray<int64_t>> filter_demand_matrix_vector;

    for (int i = 0; i < num_pe; i++) {
        int basic_iter = arr_col - 1 + ifmap_op_mat.shape()[1];
        filter_demand_matrix = xt::ones<int64_t>({basic_iter * col_fold * row_fold, arr_row}) * -1;
        cout << "filter_demand_matrix shape is " << filter_demand_matrix.shape()[0] << ", " << filter_demand_matrix.shape()[1] << endl;
        cout << filter_demand_matrix << endl;
        filter_demand_matrix_vector.push_back(filter_demand_matrix);
    }
    return filter_demand_matrix_vector;
}

vector<xt::xarray<int64_t>> SystolicPoolOs::get_ofmap_demand_matrices()
{
    cout << "get_ofmap_demand_matrices()" << endl;

    vector<xt::xarray<int64_t>> ofmap_demand_matrix_vector;

    for (int i = 0; i < num_pe; i++) {
         int inter_fold_gap_suffix = T - 1;
        xt::xarray<int64_t> inter_fold_gap_prefix_mat = xt::ones<int64_t>({inter_fold_gap_suffix, arr_col}) * -1;

        // int basic_iter = arr_row - 1 + arr_col - 1 + ifmap_op_mat.shape()[1];
        int basic_iter = arr_col - 1 + ifmap_op_mat.shape()[1];
        ofmap_demand_matrix = xt::ones<int64_t>({basic_iter * col_fold * row_fold, arr_row});

        for (int fc = 0; fc < col_fold; fc++)
        {
            for (int fr = 0; fr < row_fold; fr++)
            {
                int index = fc * row_fold * num_pe + i * row_fold + fr;
                
                // cout << "input index " << index << " of " << col_fold * row_fold << endl;
                int row_start_id = fr * arr_row;
                int row_end_idx = min(row_start_id + arr_row, Sr);
                int row_delta = arr_row - (row_end_idx - row_start_id);

                int col_start_id = fc * arr_col;
                int col_end_idx = min(col_start_id + arr_col, Sc);
                int col_delta = arr_col - (col_end_idx - col_start_id);

                xt::xarray<int64_t> this_fold_demand = xt::view(ofmap_op_mat, xt::range(row_start_id, row_end_idx), xt::range(col_start_id, col_end_idx));
                ofmap_writes += ofmap_op_mat.shape()[0] * ofmap_op_mat.shape()[1];

                if (col_delta > 0)
                {
                    xt::xarray<int64_t> null_req_mat = xt::ones<int64_t>({(int)this_fold_demand.shape()[0], col_delta}) * -1;
                    this_fold_demand = xt::concatenate(xtuple(this_fold_demand, null_req_mat), 1);
                }

                if (row_delta > 0)
                {
                    xt::xarray<int64_t> null_req_mat = xt::ones<int64_t>({row_delta, arr_col}) * -1;
                    this_fold_demand = xt::concatenate(xtuple(this_fold_demand, null_req_mat), 0);
                }

                this_fold_demand = xt::flip(this_fold_demand, 0);
                ofmap_writes += this_fold_demand.shape()[0] + this_fold_demand.shape()[1];
                this_fold_demand = xt::concatenate(xtuple(inter_fold_gap_prefix_mat, this_fold_demand), 0);


                xt::view(ofmap_demand_matrix, xt::range(basic_iter * index, basic_iter * (index + 1)), xt::all()) = this_fold_demand;
            }
        }
        cout << "ofmap_demand_matrix shape is " << ofmap_demand_matrix.shape()[0] << ", " << ofmap_demand_matrix.shape()[1] << endl;
        cout << ofmap_demand_matrix << endl;
        ofmap_demand_matrix_vector.push_back(ofmap_demand_matrix);
    }
    return ofmap_demand_matrix_vector;
}


xt::xarray<int64_t> SystolicPoolOs::skew_matrix(xt::xarray<int64_t> input_matrix_np)
{
    int rows = input_matrix_np.shape()[0];
    int cols = input_matrix_np.shape()[1];

    // std::cout << "rows: " << rows << std::endl;
    // std::cout << "cols: " << cols << std::endl;

    xt::xarray<int64_t> out_matrix_np = xt::ones<int64_t>({1, 1});
    for (int c = 0; c < cols; c++)
    {
        // std::cout << "c: " << c << std::endl;
        if (c == 0)
        {
            xt::xarray<int64_t> down_padding = -1 * xt::ones<int64_t>({cols - 1, 1});
            xt::xarray<int64_t> mat_col = xt::view(input_matrix_np, xt::all(), c);
            mat_col.reshape({rows, 1});
            out_matrix_np = xt::concatenate(xtuple(mat_col, down_padding), 0);
        }

        else
        {
            if (c == cols - 1)
            {
                xt::xarray<int64_t> up_padding = -1 * xt::ones<int64_t>({cols - 1, 1});
                xt::xarray<int64_t> mat_col = xt::view(input_matrix_np, xt::all(), c);
                mat_col.reshape({rows, 1});

                xt::xarray<int64_t> this_col = xt::concatenate(xtuple(up_padding, mat_col), 0);
                out_matrix_np = xt::concatenate(xtuple(out_matrix_np, this_col), 1);
            }
            else
            {
                xt::xarray<int64_t> up_padding = -1 * xt::ones<int64_t>({c, 1});
                xt::xarray<int64_t> mat_col = xt::view(input_matrix_np, xt::all(), c);
                mat_col.reshape({rows, 1});
                xt::xarray<int64_t> down_padding = -1 * xt::ones<int64_t>({cols - c - 1, 1});

                xt::xarray<int64_t> this_col = xt::concatenate(xtuple(up_padding, mat_col, down_padding), 0);
                out_matrix_np = xt::concatenate(xtuple(out_matrix_np, this_col), 1);
            }
        }
    }

    return out_matrix_np;
}

float SystolicPoolOs::get_avg_mapping_efficiency()
{
    float agg = 0.0f;
    int num = mapping_efficiency_per_fold.size();

    for (int i = 0; i < num; i++)
        agg += mapping_efficiency_per_fold[i];

    float avg_mapping_eff = agg / num;
    return avg_mapping_eff;
}

float SystolicPoolOs::get_avg_compute_utilization()
{
    float agg = 0.0f;
    int num = compute_utility_per_fold.size();

    for (int i = 0; i < num; i++)
        agg += compute_utility_per_fold[i];

    float avg_mapping_eff = agg / num;
    return avg_mapping_eff;
}

#endif