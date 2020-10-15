//
// Created by kylebd99 on 10/6/20.
//

#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>

std::vector<double> get_pmf(const double zipf, const size_t num_negatives) {
    std::vector<double> pmf(num_negatives);
    double harmonic_sum = 0;
    for (size_t idx = 0; idx < num_negatives; idx++) {
        pmf[idx] = 1. / pow(idx + 1, zipf);
        harmonic_sum += pmf[idx];
    }
    for (size_t idx = 0; idx < num_negatives; idx++) {
        pmf[idx] /= harmonic_sum;
    }
    return pmf;
}


int main(int arg_num, char **args) {
    std::vector<double> pmf = get_pmf(1, 100000000);
    constexpr int kReps = 20;
    double error = 0;
    double rel_error = 0;
    for (int i = 0; i < kReps; i++) {
        double exact_psi = 0;
        double exact_size = 0;
        double approx_psi = 0;
        double approx_size = 0;
        constexpr uint64_t kQueriesSeen = 1000000;
        constexpr uint64_t kPMFSampleSize = 100;
        std::vector<double> pmf_sample(kPMFSampleSize);
        std::mt19937_64 generator(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        std::uniform_int_distribution<uint64_t> distribution(0, pmf.size());
        for (uint64_t sample_idx = 0; sample_idx < kPMFSampleSize; sample_idx++)
            pmf_sample.push_back(pmf[distribution(generator)]);
        for (const auto &p : pmf_sample) {
            approx_psi += p * pow(1 - p, kQueriesSeen);
            approx_size += 1 - pow(1 - p, kQueriesSeen);
        }
        approx_psi = 1 - approx_psi * pmf.size() / static_cast<double>(kPMFSampleSize);
        approx_size = approx_size * pmf.size() / static_cast<double>(kPMFSampleSize);

        for (const auto &p : pmf) {
            exact_psi += p * pow(1 - p, kQueriesSeen);
            exact_size += 1 - pow(1 - p, kQueriesSeen);
        }
        exact_psi = 1 - exact_psi;

        std::cout << "Psi Error: " << exact_psi - approx_psi << std::endl;
        std::cout << "Relative Psi Error: " << std::abs(exact_psi - approx_psi)/exact_psi << std::endl;
        std::cout << "Size Error: " << exact_size - approx_size << std::endl;
        error += std::abs(exact_psi - approx_psi);
        rel_error += std::abs(exact_psi - approx_psi)/exact_psi;
    }
    error /= kReps;
    rel_error /= kReps;
    std::cout << "Average Error: " << error << std::endl;
    std::cout << "Average Relative Error: " << rel_error << std::endl;
}