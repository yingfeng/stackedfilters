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
#include <set>
#include <tclap/CmdLine.h>
#include "../Headers/BloomFilter.h"
#include "../Headers/StackedAMQ.h"
#include "../Headers/powerlaw.h"

power_law_distribution_t *distribution;


void divide_positives_and_negatives(std::vector<IntElement> &positives, std::vector<IntElement> &negatives,
                                    std::default_random_engine &rgen) {
    for (int i = 0; i < 100000; i++) {
        if ((rgen() % 10) == 0) {
            positives.emplace_back(i);
        } else {
            negatives.emplace_back(i);
        }
    }
}

void GenerateDataForOneRun(std::ofstream &file_stream, const double zipf_parameter, const uint64 negative_universe_size,
                           const double bits_per_positive_element, const double fpr,
                           const bool equal_fprs,
                           uint32 num_layers, const uint32 num_reps, const uint64 max_known_negatives,
                           const uint64 num_positives, const unsigned long negative_sample_size) {
    double known_fpr = 0;
    double unknown_fpr = 0;
    double total_fpr = 0;
    double used_bits = 0;
    double checks_per_pos = 0;
    double checks_per_neg = 0;
    double construction_time = 0;
    double positive_lookup_time = 0;
    double negative_lookup_time = 0;
    uint64 number_of_chosen_negatives = 0;
    const bool fixed_fpr = fpr > 0;
    std::default_random_engine rgen;
    std::uniform_int_distribution<int> generator(0, 100000);
    distribution = power_law_initialize(.5, 100000, 100000);
    std::vector<IntElement> positives;
    std::vector<IntElement> negatives;
    divide_positives_and_negatives(positives, negatives, rgen);
    const uint64 num_known_negative_elements = 20000;
    std::vector<IntElement> known_negatives(negatives.begin(), negatives.begin() + num_known_negative_elements);
    std::set<int> known_negatives_set;
    for (auto element : known_negatives) known_negatives_set.insert(element.value);
    uint64 num_positive_elements = positives.size();
    std::set<int> positives_set;
    for (auto element : positives) positives_set.insert(element.value);
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, zipf_parameter =%f, negative_universe_size=%ld, bits = %f, equal_fprs=%d, num_layers=%d\n",
           positives.size(), zipf_parameter, negative_universe_size, bits_per_positive_element, equal_fprs, num_layers);

    // Approximate Psi
    double psi = 0;
    int pos_count = 0;
    for (int i = 0; i < 100000; i++) {
        unsigned long value = power_law_simulate(generator(rgen), distribution);
        if (known_negatives_set.count(value) > 0) {
            psi++;
        }
        if (positives_set.count(value) > 0) {
            pos_count++;
        }
    }
    psi /= 100000 - pos_count;

    // Run Experiments
    for (uint32 reps = 0; reps < num_reps; reps++) {
        positives.resize(0);
        negatives.resize(0);
        known_negatives_set.clear();
        positives_set.clear();
        divide_positives_and_negatives(positives, negatives, rgen);
        known_negatives = std::vector<IntElement>(negatives.begin(), negatives.begin() + num_known_negative_elements);
        for (auto &element : known_negatives) known_negatives_set.insert(element.value);
        num_positive_elements = positives.size();
        for (auto &element : positives) positives_set.insert(element.value);
        total_size = num_positive_elements * bits_per_positive_element;
        printf("num_positive_elements =%ld, zipf_parameter =%f, negative_universe_size=%ld, bits = %f, equal_fprs=%d, num_layers=%d\n",
               positives.size(), zipf_parameter, negative_universe_size, bits_per_positive_element, equal_fprs,
               num_layers);

        // Approximate Psi
        psi = 0;
        pos_count = 0;
        for (int i = 0; i < 100000; i++) {
            unsigned long value = power_law_simulate(generator(rgen), distribution);
            if (known_negatives_set.count(value) > 0) {
                psi++;
            }
            if (positives_set.count(value) > 0) {
                pos_count++;
            }
        }
        psi /= 100000 - pos_count;
        auto construction_start = std::chrono::system_clock::now();
        // Build the Stacked Filter with the test data and layer fprs.
        StackedAMQ<BloomFilter, IntElement> filter(num_layers, positives, known_negatives, total_size, psi, .0001,
                                                   false);
        auto construction_end = std::chrono::system_clock::now();

        const auto construction_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                construction_end - construction_start);
        construction_time += construction_rep_time.count() / 1000000.0;

        used_bits += filter.GetSize() / (double) num_positive_elements;
        filter.PrintLayerDiagnostics();
        filter.ResetNumFilterChecks();

        auto positives_lookup_start = std::chrono::system_clock::now();
        // Ensure the zero false negative rate of the stacked filter.
        for (uint64 i = 0; i < num_positive_elements; i++)
            if (!filter.LookupElement(positives[i])) printf("ERROR POSITIVE REJECTED!\n");
        auto positives_lookup_end = std::chrono::system_clock::now();

        const auto postive_lookup_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                positives_lookup_end - positives_lookup_start);
        positive_lookup_time += postive_lookup_rep_time.count() / 1000000.0;

        checks_per_pos += (double) filter.NumFilterChecks() / num_positive_elements;
        filter.ResetNumFilterChecks();
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
            int value = power_law_simulate(generator(rgen), distribution);
            if (positives_set.count(value) == 0) {
                negatives_to_test.emplace_back(value);
            }
        }

        long effective_negative_sample_size = negatives_to_test.size();
        auto negative_lookup_start = std::chrono::system_clock::now();
        for (uint64 i = 0; i < effective_negative_sample_size; i++) {
            false_positives += filter.LookupElement(negatives_to_test[i]);
        }
        auto negative_lookup_end = std::chrono::system_clock::now();

        const auto negative_lookup_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                negative_lookup_end - negative_lookup_start);
        negative_lookup_time += negative_lookup_rep_time.count() / 1000000.0;

        checks_per_neg +=
                (double) filter.NumFilterChecks() / effective_negative_sample_size;
        filter.ResetNumFilterChecks();


        known_fpr += (double) (known_false_positives) /
                     (double) (std::min<uint64>(num_known_negative_elements, effective_negative_sample_size));
        unknown_fpr +=
                (double) (unknown_false_positives) / (double) (effective_negative_sample_size);
        total_fpr += (double) false_positives / (double) effective_negative_sample_size;
        psi += (double) known_negatives_tested / effective_negative_sample_size;
        printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f\n",
               (double) false_positives / (double) effective_negative_sample_size,
               num_known_negative_elements, psi);
    }
    known_fpr /= num_reps;
    unknown_fpr /= num_reps;
    total_fpr /= num_reps;
    used_bits /= num_reps;
    checks_per_neg /= num_reps;
    checks_per_pos /= num_reps;
    number_of_chosen_negatives /= num_reps;
    psi /= num_reps;
    construction_time /= num_reps;
    negative_lookup_time /= num_reps;
    positive_lookup_time /= num_reps;

    file_stream << num_positive_elements << "," <<
                negative_sample_size << "," << zipf_parameter << "," << negative_universe_size << ","
                << max_known_negatives << ","
                << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << equal_fprs << ","
                << num_layers << "," << used_bits << "," << total_fpr << ","
                << known_fpr << "," << unknown_fpr << "," << construction_time << "," << positive_lookup_time
                << ","
                << negative_lookup_time << "," << checks_per_pos << ","
                << checks_per_neg << "," << 0 << "\n";
    printf(", fpr=%f, pos_checks=%f, neg_checks=%f, construction_time=%f, positive_lookup_time=%f, negative_lookup_time=%f, EFPB=%f",
           total_fpr,
           checks_per_pos, checks_per_neg, construction_time, positive_lookup_time, negative_lookup_time,
           0);
    printf(" Used Bits= %f\n", used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, double &fpr_begin,
               double &fpr_min, double &fpr_ratio, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_known_negatives,
               uint64 &max_known_negatives_max, double &max_known_negatives_ratio, bool &equal_fprs, uint32 &num_layers,
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
    TCLAP::ValueArg<double> fpr_begin_arg("", "fpr",
                                          "The starting value for the fpr of the stacked filter, and the only value if fpr_max is not set.",
                                          false, -1, "Expects a float.");
    TCLAP::ValueArg<double> fpr_min_arg("", "fpr_min",
                                        "The min value for the fpr of the Stacked Filter.",
                                        false, .0000001, "Expects a float.");
    TCLAP::ValueArg<double> fpr_ratio_arg("", "fpr_ratio",
                                          "The step ratio over which [fpr, fpr_min) will be explored.",
                                          false, .0000000001, "Expects a float.");
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
    TCLAP::ValueArg<bool> equal_fprs_arg("", "equal_fprs",
                                         "Whether all layer fprs should be made equal.",
                                         false, false, "Expects a bool.");
    TCLAP::ValueArg<uint32> num_layers_arg("", "num_layers",
                                           "The starting value for the zipf parameter, and the only value if -zipf_end is not set.",
                                           false, 0, "Expects a uint32.");
    TCLAP::ValueArg<uint32> num_reps_arg("", "num_reps",
                                         "The number of repetitions per set of parameters.",
                                         false, 25, "Expects a uint32.");
    TCLAP::ValueArg<std::string> file_path_arg("", "path",
                                               "The path to the output file, relative or absolute.",
                                               false, "Data/TestData.csv", "Expects a string.");
    cmdLine.add(zipf_arg);
    cmdLine.add(zipf_max_arg);
    cmdLine.add(zipf_step_arg);
    cmdLine.add(bits_begin_arg);
    cmdLine.add(bits_max_arg);
    cmdLine.add(bits_step_arg);
    cmdLine.add(fpr_begin_arg);
    cmdLine.add(fpr_min_arg);
    cmdLine.add(fpr_ratio_arg);
    cmdLine.add(neg_universe_size_arg);
    cmdLine.add(neg_universe_size_max_arg);
    cmdLine.add(neg_universe_size_ratio_arg);
    cmdLine.add(max_known_negatives_arg);
    cmdLine.add(max_known_negatives_max_arg);
    cmdLine.add(max_known_negatives_ratio_arg);
    cmdLine.add(equal_fprs_arg);
    cmdLine.add(num_layers_arg);
    cmdLine.add(num_reps_arg);
    cmdLine.add(file_path_arg);
    cmdLine.parse(arg_num, args);

    zipf_begin = zipf_arg.getValue();
    zipf_max = zipf_max_arg.getValue();
    zipf_step = zipf_step_arg.getValue();
    bits_begin = bits_begin_arg.getValue();
    bits_max = bits_max_arg.getValue();
    bits_step = bits_step_arg.getValue();
    fpr_begin = fpr_begin_arg.getValue();
    fpr_min = fpr_min_arg.getValue();
    fpr_ratio = fpr_ratio_arg.getValue();
    neg_universe_size = neg_universe_size_arg.getValue();
    neg_universe_size_max = neg_universe_size_max_arg.getValue();
    neg_universe_size_ratio = neg_universe_size_ratio_arg.getValue();
    max_known_negatives = max_known_negatives_arg.getValue();
    max_known_negatives_max = max_known_negatives_max_arg.getValue();
    max_known_negatives_ratio = max_known_negatives_ratio_arg.getValue();
    equal_fprs = equal_fprs_arg.getValue();
    num_layers = num_layers_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio, neg_universe_size_ratio, max_known_negatives_ratio, num_positives_ratio;
    uint64 num_positives_begin, num_positives_max, neg_universe_size_begin, neg_universe_size_max, max_known_negatives_begin, max_known_negatives_max;
    uint32 num_reps, num_layers;
    std::string file_path;
    bool equal_fprs;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, max_known_negatives_begin, max_known_negatives_max, max_known_negatives_ratio,
              equal_fprs, num_layers,
              num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Number Of Positive Elements,Negative Sample Size,Zipf Parameter,Negative Universe Size,Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
               "FPR,Known FPR,Unknown FPR,Construction Time,Positive Lookup Time,Negative Lookup Time,Filter Checks For Positive,Filter Checks For "
               "Negative,EFPB\n";
    const uint64 num_positives = 1000000;
    const uint64 negative_sample_size = 1000000;
    for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
        if (zipf_parameter <= 1.01 && zipf_parameter >= .99) continue;
        for (double neg_universe_size = neg_universe_size_begin;
             neg_universe_size < neg_universe_size_max; neg_universe_size *= neg_universe_size_ratio) {
            for (uint64 max_known_negatives = max_known_negatives_begin;
                 max_known_negatives <= max_known_negatives_max; max_known_negatives *= max_known_negatives_ratio) {
                if (fpr_begin < 0) {
                    for (double bits_per_positive_element = bits_begin;
                         bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                        GenerateDataForOneRun(file_stream, zipf_parameter,
                                              neg_universe_size,
                                              bits_per_positive_element,
                                              fpr_begin,
                                              equal_fprs,
                                              num_layers, num_reps, max_known_negatives, num_positives,
                                              negative_sample_size);
                        file_stream.flush();
                    }
                } else {
                    for (double fpr = fpr_begin;
                         fpr >= fpr_min; fpr *= fpr_ratio) {
                        GenerateDataForOneRun(file_stream, zipf_parameter,
                                              neg_universe_size,
                                              bits_begin,
                                              fpr,
                                              equal_fprs,
                                              num_layers, num_reps, max_known_negatives, num_positives,
                                              negative_sample_size);
                        file_stream.flush();
                    }

                }
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop