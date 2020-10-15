#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
//
// Created by kylebd99 on 8/30/19.
//

#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <chrono>
#include <tclap/CmdLine.h>
#include "../Headers/cuckoofilter.h"
#include "../Headers/StackedAMQ.h"
#include "../Headers/ZipfDistribution.h"

std::vector<IntElement> generate_ints(uint64 num_elements) {
    std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
    std::vector<IntElement> int_vec(num_elements);
    for (uint64 i = 0; i < num_elements; i++) {
        int_vec[i] = i;
    }
    std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
    return int_vec;
}

int main(int arg_num, char **args) {
    uint64 nslots = pow(2, 16);
    uint key_remainder = 10;
    uint quotient = ceil(log2(nslots));
    cuckoofilter::CuckooFilter<10> filter(nslots, 123456);
    for (uint32 i = 0; i < 9500; i++)
        filter.Add(i);
    for (uint32 i = 0; i < 9500; i++)
        printf("Should Contain: %d\n", filter.Contain(i) == cuckoofilter::Status::Ok);
    for (uint32 i = 0; i < 9500; i++)
        printf("Shouldnt Contain: %d\n", filter.Contain(i + 9500) == cuckoofilter::Status::Ok);
}

#pragma clang diagnostic pop