#ifndef _operand_matrix_h
#define _operand_matrix_h

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "../scale_config.h"
#include "../topology_utils.h"

class OperandMatrix
{
public:
    OperandMatrix();
    ~OperandMatrix();
    void set_params(Config *config, Topology *topology, int64_t layer_id);
    xt::xarray<int64_t>& get_ifmap_matrix();
    xt::xarray<int64_t>& get_filter_matrix();
    xt::xarray<int64_t>& get_ofmap_matrix();

private:
    void create_ifmap_matrix();
    void create_filter_matrix();
    void create_ofmap_matrix();
    void create_operand_matrices();

    int64_t calc_ifmap_elem_addr(int64_t i, int64_t j, int input_index);
    int64_t calc_filter_elem_addr(int64_t i, int64_t j);
    int64_t calc_ofmap_elem_addr(int64_t i, int64_t j);

    Config *config;
    Topology *topology;
    int64_t layer_id;

    int64_t ifmap_rows;
    int64_t ifmap_cols;
    int64_t filter_rows;
    int64_t filter_cols;
    int64_t num_input_channels;
    int64_t num_filters;
    int64_t row_stride;
    int64_t col_stride;

    int64_t batch_size;
    int64_t word_size;

    int64_t ofmap_rows;
    int64_t ofmap_cols;
    int64_t ofmap_px_per_filt;
    int64_t conv_window_size;

    vector<int64_t> ifmap_offset;
    int64_t filter_offset;
    int64_t ofmap_offset;

    xt::xarray<int64_t> ifmap_addr_matrix;
    xt::xarray<int64_t> filter_addr_matrix;
    xt::xarray<int64_t> ofmap_addr_matrix;

    bool params_set_flag = false;
    bool matrices_ready_flag = false;
};

OperandMatrix::OperandMatrix()
{
}

OperandMatrix::~OperandMatrix()
{
    // delete(ifmap_addr_matrix.data());
    // delete(filter_addr_matrix.data());
    // delete(ofmap_addr_matrix.data());
}

void OperandMatrix::set_params(Config *config, Topology *topology, int64_t layer_id)
{
    this->config = config;
    this->topology = topology;
    this->layer_id = layer_id;

    this->ifmap_rows = this->topology->get_layer_ifmap_dims(this->layer_id).first;
    this->ifmap_cols = this->topology->get_layer_ifmap_dims(this->layer_id).second;

    this->filter_rows = this->topology->get_layer_filter_dims(this->layer_id).first;
    this->filter_cols = this->topology->get_layer_filter_dims(this->layer_id).second;

    this->num_input_channels = this->topology->get_layer_num_channels(this->layer_id);
    this->num_filters = this->topology->get_layer_num_filters(this->layer_id);

    this->row_stride = this->topology->get_layer_strides(this->layer_id).first;
    this->col_stride = this->topology->get_layer_strides(this->layer_id).second;

    this->batch_size = this->config->get_batch_size();
    this->word_size = this->config->get_word_size();

    this->ofmap_rows = this->topology->get_layer_ofmap_dims(this->layer_id).first;
    this->ofmap_cols = this->topology->get_layer_ofmap_dims(this->layer_id).second;
    this->ofmap_px_per_filt = this->ofmap_rows * this->ofmap_cols;
    this->conv_window_size = this->topology->get_layer_window_size(this->layer_id);

    auto offsetinfo = this->topology->get_layer_offsets(this->layer_id);
    this->ifmap_offset = offsetinfo.ifmap_offset;
    this->filter_offset = offsetinfo.filter_offset;
    this->ofmap_offset = offsetinfo.ofmap_offset;

    this->ifmap_addr_matrix = xt::ones<int64_t>({this->ofmap_px_per_filt * this->batch_size, this->conv_window_size});
    this->filter_addr_matrix = xt::ones<int64_t>({this->conv_window_size, this->num_filters});
    this->ofmap_addr_matrix = xt::ones<int64_t>({this->ofmap_px_per_filt * this->batch_size, this->num_filters});

    for (int i = 0; i < this->ifmap_offset.size(); i++) {
        cout << "ifmap_offset is " << this->ifmap_offset[i] << ", ";
    }
    cout << "filter_offset is " << this->filter_offset;
    cout << ", ofmap_offset is " << this->ofmap_offset << endl;
    // cout << ", ifmap:filter ratio is " << ratio << endl;

    params_set_flag = true;
}

xt::xarray<int64_t>& OperandMatrix::get_ifmap_matrix()
{
    if (!matrices_ready_flag)
        create_operand_matrices();
    return ifmap_addr_matrix;
}

xt::xarray<int64_t>& OperandMatrix::get_filter_matrix()
{
    if (!matrices_ready_flag)
        create_operand_matrices();
    return filter_addr_matrix;
}

xt::xarray<int64_t>& OperandMatrix::get_ofmap_matrix()
{
    if (!matrices_ready_flag)
        create_operand_matrices();
    return ofmap_addr_matrix;
}

void OperandMatrix::create_operand_matrices()
{
    if (!params_set_flag)
        cout << "Parameters not set yet. Run set_params(). Exiting!!!" << endl;
    create_ifmap_matrix();
    create_filter_matrix();
    create_ofmap_matrix();
    matrices_ready_flag = true;
}

void OperandMatrix::create_ifmap_matrix()
{
    uint64_t per_input_size = batch_size * ofmap_px_per_filt / ifmap_offset.size();
    for (int64_t row_idx = 0; row_idx < batch_size * ofmap_px_per_filt; row_idx++)
    {
        for (int64_t col_idx = 0; col_idx < conv_window_size; col_idx++)
        {
            ifmap_addr_matrix(row_idx, col_idx) = calc_ifmap_elem_addr(row_idx, col_idx, row_idx / per_input_size);
        }
    }
}

void OperandMatrix::create_filter_matrix()
{
    for (int64_t row_idx = 0; row_idx < conv_window_size; row_idx++)
    {
        for (int64_t col_idx = 0; col_idx < num_filters; col_idx++)
        {
            filter_addr_matrix(row_idx, col_idx) = calc_filter_elem_addr(row_idx, col_idx);
        }
    }
}

void OperandMatrix::create_ofmap_matrix()
{
    for (int64_t row_idx = 0; row_idx < batch_size * ofmap_px_per_filt; row_idx++)
    {
        for (int64_t col_idx = 0; col_idx < num_filters; col_idx++)
        {
            ofmap_addr_matrix(row_idx, col_idx) = calc_ofmap_elem_addr(row_idx, col_idx);
        }
    }
}

int64_t OperandMatrix::calc_ifmap_elem_addr(int64_t i, int64_t j, int input_index)
{
    int64_t offset = ifmap_offset[input_index];
    int64_t filter_col = filter_cols;
    int64_t r_stride = row_stride;
    int64_t c_stride = col_stride;
    int64_t Ew = ofmap_cols;
    int64_t channel = num_input_channels;

    int64_t ofmap_row = i / Ew;
    int64_t ofmap_col = i % Ew;

    int64_t i_row = ofmap_row * r_stride;
    int64_t i_col = ofmap_col * c_stride;

    int64_t window_addr = i_row * ifmap_cols * channel + i_col * channel;

    int64_t c_row = j / (filter_col * channel);
    int64_t k = j % (filter_col * channel);
    int64_t c_col = k / channel;
    int64_t c_ch = k % channel;
    int64_t ifmap_px_addr = -1;

    // cout << c_row << ", " << i_row << ", " << ifmap_rows << endl;
    // cout << c_col << ", " << i_col << ", " << ifmap_cols << endl;

    if ((c_row + i_row >= ifmap_rows * batch_size) || (c_col + i_col >= ifmap_cols))
    {
        ifmap_px_addr = -1;
    }
    else
    {
        int64_t internal_address = c_row * (ifmap_cols * channel) + c_col * channel + c_ch;
        ifmap_px_addr = (internal_address + window_addr) * word_size + offset;
    }
    return ifmap_px_addr;
}

int64_t OperandMatrix::calc_filter_elem_addr(int64_t i, int64_t j)
{
    int64_t offset = filter_offset;
    int64_t filter_row = filter_rows;
    int64_t filter_col = filter_cols;
    int64_t channel = num_input_channels;
    int64_t internal_address = j * filter_row * filter_col * channel + i;
    int64_t filter_px_addr = internal_address * word_size + offset;
    return filter_px_addr;
}

int64_t OperandMatrix::calc_ofmap_elem_addr(int64_t i, int64_t j)
{
    int64_t offset = ofmap_offset;
    int64_t num_filt = num_filters;
    int64_t internal_address = num_filt * i + j;
    int64_t ofmap_px_addr = internal_address * word_size + offset;

    // if (ofmap_px_addr >= 101126815 && ofmap_px_addr < 101312256)
    // {
    //     cout << "ofmap_offset is " << ofmap_offset << endl;
    //     cout << "calc_ofmap_elem_addr " << ofmap_px_addr << endl;
    // }

    return ofmap_px_addr;
}

#endif