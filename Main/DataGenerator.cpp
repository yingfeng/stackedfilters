//
// Created by kylebd99 on 8/30/19.
//

#include <cstdlib>
#include <fstream>
#include <random>
#include "../Headers/BloomFilter.h"
#include "../Headers/StackedAMQ.h"

std::vector<IntElement> generate_ints(uint64 num_elements) {
    /* Distribution on which to apply the generator */
    std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
    std::vector<IntElement> int_vec(num_elements);
    for (uint64 i = 0; i < num_elements; i++) {
        int_vec[i] = i;
    }
    std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
    return int_vec;
}


void GenerateDataForOneRun(std::ofstream &file_stream, const double beta, const double psi, const double bits,
                           const bool equal_fprs,
                           const int num_layers, const int num_reps, const uint64 total_elements) {

    int num_positives = total_elements * beta;
    int num_known_negatives = total_elements * (1 - beta);
    int num_unknown_negatives = num_known_negatives;
    int total_size = num_positives * bits;
    printf("beta =%f, psi=%f, bits = %f, equal_fprs=%d, num_layers=%d\n",
           beta, psi, bits, equal_fprs, num_layers);
    double known_fpr = 0;
    double unknown_fpr = 0;
    double total_fpr = 0;
    double used_bits = 0;
    double checks_per_pos = 0;
    double checks_per_neg = 0;
    for (int reps = 0; reps < num_reps; reps++) {
        std::vector<IntElement> ints = generate_ints(
                num_positives + num_known_negatives + num_unknown_negatives);
        std::vector<IntElement> positives = std::vector<IntElement>(
                ints.begin(), ints.begin() + num_positives);
        std::vector<IntElement> known_negatives = std::vector<IntElement>(
                ints.begin() + num_positives,
                ints.begin() + num_positives + num_known_negatives);
        std::vector<IntElement> unknown_negatives = std::vector<IntElement>(
                ints.begin() + num_positives + num_known_negatives, ints.end());
        StackedAMQ<BloomFilter, IntElement> filter(
                num_layers, positives, known_negatives, total_size, psi,
                0, equal_fprs);
        filter.PrintLayerDiagnostics();
        filter.ResetNumFilterChecks();
        for (int i = 0; i < num_positives; i++)
            if (!filter.LookupElement(positives[i])) printf("ERROR POSITIVE REJECTED!\n");
        checks_per_pos += (double) filter.NumFilterChecks() / num_positives;
        filter.ResetNumFilterChecks();
        int known_false_positives = 0;
        for (int i = 0; i < num_known_negatives; i++) {
            if (filter.LookupElement(known_negatives[i])) {
                known_false_positives++;
            }
        }
        checks_per_neg +=
                (double) filter.NumFilterChecks() / num_known_negatives * psi;
        filter.ResetNumFilterChecks();
        int unknown_false_positives = 0;
        for (int i = 0; i < num_unknown_negatives; i++) {
            if (filter.LookupElement(unknown_negatives[i])) {
                unknown_false_positives++;
            }
        }
        checks_per_neg += (double) filter.NumFilterChecks() /
                          num_unknown_negatives * (1 - psi);
        known_fpr += (double) (known_false_positives) / (double) (num_known_negatives);
        unknown_fpr += (double) (unknown_false_positives) / (double) (num_unknown_negatives);
        total_fpr +=
                psi * (double) (known_false_positives) / (double) (num_known_negatives) +
                (1 - psi) * (double) (unknown_false_positives) / (double) (num_unknown_negatives);
        used_bits += (double) filter.GetSize() / num_positives;
        printf("Trial FPR:%f\n",
               psi * (double) (known_false_positives) / (double) (num_known_negatives) +
               (1 - psi) * (double) (unknown_false_positives) /
               (double) (num_unknown_negatives));
    }
    known_fpr = known_fpr / num_reps;
    unknown_fpr = unknown_fpr / num_reps;
    total_fpr = total_fpr / num_reps;
    used_bits = used_bits / num_reps;
    checks_per_neg = checks_per_neg / num_reps;
    checks_per_pos = checks_per_pos / num_reps;
    file_stream << beta << "," << psi << "," << bits << "," << equal_fprs << ","
                << num_layers << "," << used_bits << "," << total_fpr << ","
                << known_fpr << "," << unknown_fpr << "," << checks_per_pos << ","
                << checks_per_neg << "\n";
    printf(", fpr=%f, pos_checks=%f, neg_checks=%f", total_fpr,
           checks_per_pos, checks_per_neg);
    printf(" Used Bits= %f\n", used_bits);
}


int main(int arg_num, char **args) {
    std::ofstream file_stream;
    file_stream.open("Data/Stacked_Filter_Data.csv");
    file_stream << "Beta,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
                   "FPR,Known FPR,UnknownFPR,Filter Checks For Positive,Filter Checks For "
                   "Negative\n";
    uint64 total_elements = 300000;
    uint num_reps = 3;
    for (double beta = .8; beta < .82; beta += .04) {
        for (double psi = .8; psi < .82; psi += .04) {
            for (double bits_per_element = 12; bits_per_element < 14; bits_per_element += 4) {
                for (int num_layers = 1; num_layers <= 3; num_layers += 2) {
                    bool equal_fprs = false;
                    GenerateDataForOneRun(file_stream, beta, psi, bits_per_element, equal_fprs, num_layers, num_reps,
                                          total_elements);
                }
            }
        }
    }
    file_stream.close();
}