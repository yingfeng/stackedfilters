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
#include "StackedFilter.h"
#include "ZipfDistribution.h"

std::vector<IntElement> generate_ints(uint64 num_elements) {
    std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
    std::vector<IntElement> int_vec(num_elements);
    for (uint64 i = 0; i < num_elements; i++) {
        int_vec[i] = i;
    }
    std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
    return int_vec;
}

std::vector<double> uniform_cdf(uint64_t num_elements) {
    std::vector<double> cdf;
    double sum = 0;
    for (uint64_t i = 1; i <= num_elements; i++) {
        sum += 1. / num_elements;
        cdf.push_back(sum);
    }
    return cdf;
}


int main(int arg_num, char **args) {
    static constexpr int kBitsPerElement = 10;
    static constexpr int kNumPositives = 1000;
    static constexpr int kNumNegatives = 15000;
    std::vector<IntElement> ints = generate_ints(kNumPositives + kNumNegatives);
    std::vector<IntElement> positives(ints.begin(), ints.begin() + kNumPositives);
    std::vector<IntElement> negatives(ints.begin() + kNumPositives, ints.end());
    std::vector<double> cdf = uniform_cdf(kNumNegatives);

    StackedFilter<BloomFilterLayer, IntElement> filter(kBitsPerElement * positives.size(), positives, negatives, cdf);

    double false_positives = 0;
    for(const auto& neg : negatives){
        false_positives += filter.LookupElement(neg);
    }
    std::cout << "Stacked Filter False Positive Rate: " << false_positives/negatives.size() << std::endl;
}

#pragma clang diagnostic pop