#ifndef _dram_h
#define _dram_h

class DRAM {
public:
    DRAM();
    int64_t get_latency() {return hit_latency; }
private:
    int64_t hit_latency;
};

DRAM::DRAM() {
    hit_latency = 40;
}

#endif