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
#include "BloomFilterLayer.h"
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

std::vector<double> zipf_cdf(uint64_t num_elements, double zipf_param) {
    std::vector<double> pmf;
    double sum = 0;
    for (uint64_t i = 1; i <= num_elements; i++) {
        sum += 1. / pow(i, zipf_param);
        pmf.push_back(sum);
    }
    for (auto &prob : pmf) {
        prob /= sum;
    }
    return pmf;
}

void GenerateDataForOneRun(std::ofstream &file_stream, const uint64 negative_universe_size, const double zipf_parameter,
                           const double bits_per_positive_element, const uint32 num_reps,
                           const uint64 max_known_negatives, const uint64 num_positives,
                           const uint64 negative_sample_size, const std::vector<double> &cdf) {
    std::vector<double> total_fpr(10);
    double used_bits = 0;
    uint64 number_of_chosen_negatives = 0;
    long double psi = 0;
    double insert_capacity = .25;
    double additional_inserts = insert_capacity + .25;
    uint64 total_positives = num_positives * (1 + additional_inserts);
    uint64 positives_capacity = num_positives * (1 + insert_capacity);
    std::vector<IntElement> ints = generate_ints(total_positives + negative_universe_size);
    std::vector<IntElement> starting_positives(ints.begin(), ints.begin() + num_positives);
    std::vector<IntElement> additional_positives(ints.begin() + num_positives, ints.begin() + total_positives);
    std::vector<IntElement> negatives(ints.begin() + total_positives, ints.end());
    uint64 total_size = positives_capacity * bits_per_positive_element;
    printf("num_positive_elements =%ld, zipf_parameter =%f, negative_universe_size=%ld, bits = %f, equal_fprs=%d, num_layers=%d\n",
           total_positives, zipf_parameter, negative_universe_size, bits_per_positive_element);

    for (uint32 reps = 0; reps < num_reps; reps++) {

        // Build the Stacked Filter with the test data and layer fprs.
        StackedFilter<BloomFilterLayer, IntElement> filter(bits_per_positive_element * starting_positives.size(),
                                                           starting_positives, negatives, cdf, insert_capacity);

        used_bits += filter.GetSize() / (double) positives_capacity;
        filter.PrintLayerDiagnostics();
        filter.ResetNumFilterChecks();
        uint64 num_known_negative_elements = filter.num_negative_;
        std::vector<IntElement> known_negatives = std::vector<IntElement>(
                negatives.begin(),
                negatives.begin() + num_known_negative_elements);

        // Ensure the zero false negative rate of the stacked filter.
        for (auto &starting_positive : starting_positives) {
            if (!filter.LookupElement(starting_positive)) printf("ERROR POSITIVE REJECTED!\n");
        }
        int num_steps = 10;
        uint64_t step_size = additional_positives.size() / num_steps;
        for (int i = 0; i < num_steps; i++) {
            for (uint64_t j = i * step_size; j < (i + 1) * step_size; j++) {
                filter.InsertPositiveElement(additional_positives[j]);
            }
            filter.PrintLayerDiagnostics();

            // Test the false positive rate for known negatives.
            uint64 false_positives = 0;
            uint64 known_false_positives = 0;
            uint64 unknown_false_positives = 0;
            uint64 known_negatives_tested = 0;
            uint64 unknown_negatives_tested = 0;
            std::minstd_rand gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(0, 1);
            std::vector<IntElement> negatives_to_test;
            for (uint64 i = 0; i < negative_sample_size; i++) {
                double uniform_double = dist(gen);
                uint64 element_rank = inverseCdfFast(uniform_double, zipf_parameter, negative_universe_size);
                negatives_to_test.push_back(negatives[element_rank]);
                if (element_rank < num_known_negative_elements) {
                    known_negatives_tested++;
                } else {
                    unknown_negatives_tested++;
                }
            }

            for (uint64 j = 0; j < negative_sample_size; j++) {
                false_positives += filter.LookupElement(negatives_to_test[j]);
            }

            for (uint64 j = 0; j < std::min<uint64>(num_known_negative_elements, negative_sample_size); j++) {
                known_false_positives += filter.LookupElement(known_negatives[j]);
            }

            for (uint64 j = 0; j < negative_sample_size; j++) {
                unknown_false_positives += filter.LookupElement(negatives[j + num_known_negative_elements]);
            }

            total_fpr[i] += (double) false_positives / (double) negative_sample_size;
            psi += (double) known_negatives_tested / negative_sample_size;
            printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f\n",
                   (double) false_positives / (double) negative_sample_size,
                   num_known_negative_elements, (double) known_negatives_tested / negative_sample_size);
        }
    }
    for (auto &temp_fpr : total_fpr) temp_fpr /= num_reps;
    used_bits /= num_reps;
    number_of_chosen_negatives /= num_reps;
    psi /= num_reps;

    for (unsigned int i = 0; i < total_fpr.size(); i++) {
        double percent_capacity = (1 - insert_capacity +
                                   additional_inserts * (static_cast<double>(i) / total_fpr.size()));
        file_stream << total_positives << "," <<
                    negative_sample_size << "," << zipf_parameter << "," << negative_universe_size << ","
                    << max_known_negatives << ","
                    << number_of_chosen_negatives << "," << psi << ","
                    << bits_per_positive_element << "," << used_bits << "," << total_fpr[i] << ","
                    << percent_capacity << "\n";
        std::cout << "Total FPR: " << total_fpr[i] << " Used Bits: " << used_bits << " Percent Capacity: "
                  << percent_capacity << std::endl;
    }
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_known_negatives,
               uint64 &max_known_negatives_max, double &max_known_negatives_ratio,
               uint32 &num_reps, std::string &file_path, int arg_num, char **args) {
    TCLAP::ValueArg<double> zipf_arg("", "zipf",
                                     "The starting value for the zipf parameter, and the only value if zipf_max is not set.",
                                     false, .75, "Expects a float.");
    TCLAP::ValueArg<double> zipf_max_arg("", "zipf_max",
                                         "The max value for the zipf parameter.", false, 10, "Expects a float.");
    TCLAP::ValueArg<double> zipf_step_arg("", "zipf_step",
                                          "The step size over which [zipf, zipf_max) will be explored.",
                                          false, 100, "Expects a float.");
    TCLAP::ValueArg<double> bits_begin_arg("", "bits",
                                           "The starting value for the bits per positive element, and the only value if bits_max is not set.",
                                           false, 10, "Expects a float.");
    TCLAP::ValueArg<double> bits_max_arg("", "bits_max",
                                         "The max value for the bits per positive element",
                                         false, 100, "Expects a float.");
    TCLAP::ValueArg<double> bits_step_arg("", "bits_step",
                                          "The step size over which [bits, bits_max) will be explored.",
                                          false, 1000, "Expects a float.");
    TCLAP::ValueArg<uint64> neg_universe_size_arg("", "neg_universe_size",
                                                  "The starting value for the negative universe size, and the only value if neg_universe_size_max is not set.",
                                                  false, 100000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> neg_universe_size_max_arg("", "neg_universe_size_max",
                                                      "The max value for the negative universe size.",
                                                      false, 10000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> neg_universe_size_ratio_arg("", "neg_universe_size_ratio",
                                                        "The ratio between steps over which [neg_universe_size, neg_universe_size_max) will be explored.",
                                                        false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint64> max_known_negatives_arg("", "max_known_negatives",
                                                    "The maximum number of known negatives available to the stacked filter.",
                                                    false, 10000000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> max_known_negatives_max_arg("", "max_known_negatives_max",
                                                        "The max value for the max number of negatives it can use.",
                                                        false, 100000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> max_known_negatives_ratio_arg("", "max_known_negatives_ratio",
                                                          "The ratio between steps over which [max_known_negatives, max_known_negatives_max) will be explored.",
                                                          false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint32> num_reps_arg("", "num_reps",
                                         "The number of repetitions per set of parameters.",
                                         false, 10, "Expects a uint32.");
    TCLAP::ValueArg<std::string> file_path_arg("", "path",
                                               "The path to the output file, relative or absolute.",
                                               false, "Data/TestData.csv", "Expects a string.");
    cmdLine.add(zipf_arg);
    cmdLine.add(zipf_max_arg);
    cmdLine.add(zipf_step_arg);
    cmdLine.add(bits_begin_arg);
    cmdLine.add(bits_max_arg);
    cmdLine.add(bits_step_arg);
    cmdLine.add(neg_universe_size_arg);
    cmdLine.add(neg_universe_size_max_arg);
    cmdLine.add(neg_universe_size_ratio_arg);
    cmdLine.add(max_known_negatives_arg);
    cmdLine.add(max_known_negatives_max_arg);
    cmdLine.add(max_known_negatives_ratio_arg);
    cmdLine.add(num_reps_arg);
    cmdLine.add(file_path_arg);
    cmdLine.parse(arg_num, args);

    zipf_begin = zipf_arg.getValue();
    zipf_max = zipf_max_arg.getValue();
    zipf_step = zipf_step_arg.getValue();
    bits_begin = bits_begin_arg.getValue();
    bits_max = bits_max_arg.getValue();
    bits_step = bits_step_arg.getValue();
    neg_universe_size = neg_universe_size_arg.getValue();
    neg_universe_size_max = neg_universe_size_max_arg.getValue();
    neg_universe_size_ratio = neg_universe_size_ratio_arg.getValue();
    max_known_negatives = max_known_negatives_arg.getValue();
    max_known_negatives_max = max_known_negatives_max_arg.getValue();
    max_known_negatives_ratio = max_known_negatives_ratio_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, neg_universe_size_ratio, max_known_negatives_ratio, num_positives_ratio;
    uint64 neg_universe_size_begin, neg_universe_size_max, max_known_negatives_begin, max_known_negatives_max;
    uint32 num_reps, num_layers;
    std::string file_path;
    bool equal_fprs;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, max_known_negatives_begin, max_known_negatives_max, max_known_negatives_ratio,
              num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Number Of Positive Elements,Negative Sample Size,Zipf Parameter,Negative Universe Size,Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Used Bits,Total "
               "FPR,Percent Capacity Filled\n";
    const uint64 num_positives = 1000000;
    const uint64 negative_sample_size = 1000000;
    for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
        if (zipf_parameter <= 1.01 && zipf_parameter >= .99) continue;
        for (double neg_universe_size = neg_universe_size_begin;
             neg_universe_size < neg_universe_size_max; neg_universe_size *= neg_universe_size_ratio) {
            const std::vector<double> cdf = zipf_cdf(neg_universe_size, zipf_parameter);
            for (uint64 max_known_negatives = max_known_negatives_begin;
                 max_known_negatives <= max_known_negatives_max; max_known_negatives *= max_known_negatives_ratio) {
                for (double bits_per_positive_element = bits_begin;
                     bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                    GenerateDataForOneRun(file_stream, neg_universe_size, zipf_parameter,
                                          bits_per_positive_element,
                                          num_reps, max_known_negatives, num_positives,
                                          negative_sample_size, cdf);
                    file_stream.flush();
                }
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop