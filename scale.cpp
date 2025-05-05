#include <iostream>
#include <string>

#include "scale_sim.h"

int main(int argc, char* argv[])
{
    char* topology;
    char* config;
    
    if (argc < 3) {
        topology = "./topologies/conv_nets/test.csv";
        config = "./configs/scale.cfg";
    } else {
        topology = argv[1];
        config = argv[2];
    }

    char* logpath = "./test_runs";
    char* inp_type = "conv";

    bool gemm_input = false;
    if (inp_type == "gemm")
        gemm_input = true;


    ScaleSim* scaleSim = new ScaleSim(true, config, topology, gemm_input);
    scaleSim->run_scale(logpath);

    return 0;
}