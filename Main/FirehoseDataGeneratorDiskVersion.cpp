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
                           const bool use_hdd, const bool allow_caching,
                           uint32 num_layers, const uint32 num_reps, const uint64 max_known_negatives,
                           const uint64 num_positives, const unsigned long negative_sample_size) {
    double known_fpr = 0;
    double unknown_fpr = 0;
    double total_fpr = 0;
    double used_bits = 0;
    double checks_per_pos = 0;
    double checks_per_neg = 0;
    double construction_time = 0;
    double lookup_time = 0;
    double filter_time = 0;
    double disk_time = 0;
    uint64 number_of_chosen_negatives = 0;
    const bool fixed_fpr = fpr > 0;
    std::default_random_engine rgen;
    std::uniform_int_distribution<unsigned long> generator(0, 10000000);
    distribution = power_law_initialize(.5, 100000, 10000000);
    std::vector<IntElement> positives;
    std::vector<IntElement> negatives;
    divide_positives_and_negatives(positives, negatives, rgen);
    const uint64 num_known_negative_elements = 1000;
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

        // Test the false positive rate for known negatives.
        uint64 false_positives = 0;
        uint64 known_false_positives = 0;
        uint64 unknown_false_positives = 0;
        uint64 known_negatives_tested = 0;
        uint64 unknown_negatives_tested = 0;
        // Disk Timing
        uint64_t temp_int = 0;
        uint64_t sum_int = 0;
        char *file_name;
        char hdd_file[] = "/mnt/54258bba-33d9-42ef-a1b7-1941cefb1dcd/hdd_temp.txt";
        char ssd_file[] = "/home/kylebd99/Downloads/ssd_temp.txt";
        if (use_hdd) {
            file_name = hdd_file;
        } else {
            file_name = ssd_file;
        }
        FILE *fileptr = fopen(file_name, "r");
        fseek(fileptr, 0L, SEEK_END);
        uint64_t file_size = ftell(fileptr);
        uint64_t file_pos = 0;

        // Generate the sample queries.
        std::vector<IntElement> elements_to_test;
        for (uint64 i = 0; i < negative_sample_size; i++) {
            unsigned long element_rank = power_law_simulate(generator(rgen), distribution);
            elements_to_test.push_back(negatives[element_rank]);
            if (element_rank < num_known_negative_elements) {
                known_negatives_tested++;
            } else {
                unknown_negatives_tested++;
            }
        }

        // Filter Timing
        auto filter_start = std::chrono::high_resolution_clock::now();
        for (const auto &element : elements_to_test) {
            const bool is_false_positive = filter.LookupElement(element);
            if (is_false_positive) {
                false_positives++;
            }
        }
        auto filter_end = std::chrono::high_resolution_clock::now();
        filter_time += std::chrono::duration_cast<std::chrono::microseconds>(
                filter_end - filter_start).count() / 1000000.0;

        auto disk_start = std::chrono::high_resolution_clock::now();
        for (uint64 i = 0; i < false_positives; i++) {
            if (allow_caching) {
                file_pos = (elements_to_test[i].value * 100000) % file_size;
            } else {
                sum_int += rand();
                file_pos = (sum_int +
                            negatives[(i * i + sum_int) % negatives.size()].value) * 10000 % file_size;
            }
            fseek(fileptr, file_pos, SEEK_SET);
            fread(&temp_int, sizeof(uint64_t), 1, fileptr);
            sum_int += temp_int;
        }
        auto disk_end = std::chrono::high_resolution_clock::now();
        disk_time += std::chrono::duration_cast<std::chrono::microseconds>(
                disk_end - disk_start).count() / 1000000.0;


        for (uint64 i = 0; i < std::min<uint64>(num_known_negative_elements, negative_sample_size); i++) {
            known_false_positives += filter.LookupElement(known_negatives[i]);
        }
        for (uint64 i = 0; i < negatives.size() - num_known_negative_elements; i++) {
            unknown_false_positives += filter.LookupElement(negatives[i + num_known_negative_elements]);
        }

        known_fpr += (double) (known_false_positives) /
                     (double) (std::min<uint64>(num_known_negative_elements, negative_sample_size));
        unknown_fpr +=
                (double) (unknown_false_positives) / (double) (negative_sample_size);
        total_fpr += (double) false_positives / (double) negative_sample_size;
        psi += (double) known_negatives_tested / negative_sample_size;
        printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f, sum_int:%ld\n",
               (double) false_positives / (double) negative_sample_size,
               num_known_negative_elements, (double) known_negatives_tested / negative_sample_size, sum_int);
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
    filter_time /= num_reps;
    disk_time /= num_reps;
    lookup_time = filter_time + disk_time;

    if (use_hdd) {
        filter_time *= 10;
        disk_time *= 10;
        lookup_time *= 10;
    }

    file_stream << num_positive_elements << ","
                << negative_sample_size << "," << 0 << "," << negative_universe_size << ","
                << max_known_negatives << ","
                << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << equal_fprs << ","
                << num_layers << "," << used_bits << "," << total_fpr << ","
                << known_fpr << "," << unknown_fpr << "," << construction_time
                << ","
                << lookup_time << "," << filter_time << "," << disk_time << "\n";
    printf(", fpr=%f, pos_checks=%f, neg_checks=%f, construction_time=%f, lookup_time=%f, filter_time=%f, disk_time=%f",
           total_fpr,
           checks_per_pos, checks_per_neg, construction_time, lookup_time, filter_time, disk_time);
    printf(" Used Bits= %f\n", used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, double &fpr_begin,
               double &fpr_min, double &fpr_ratio, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_known_negatives,
               uint64 &max_known_negatives_max, double &max_known_negatives_ratio, bool &equal_fprs, bool &use_hdd,
               bool &allow_caching, uint32 &num_layers,
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
    TCLAP::ValueArg<bool> use_hdd_arg("", "use_hdd",
                                      "Whether hdd should be used to read.",
                                      false, false, "Expects a bool.");
    TCLAP::ValueArg<bool> allow_caching_arg("", "allow_caching",
                                            "Whether caching effects should be minimized.",
                                            false, false, "Expects a bool.");
    TCLAP::ValueArg<uint32> num_layers_arg("", "num_layers",
                                           "The starting value for the zipf parameter, and the only value if -zipf_end is not set.",
                                           false, 3, "Expects a uint32.");
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
    cmdLine.add(use_hdd_arg);
    cmdLine.add(allow_caching_arg);
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
    use_hdd = use_hdd_arg.getValue();
    allow_caching = allow_caching_arg.getValue();
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
    bool equal_fprs, use_hdd, allow_caching;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, max_known_negatives_begin, max_known_negatives_max, max_known_negatives_ratio,
              equal_fprs, use_hdd, allow_caching, num_layers,
              num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);

    file_stream
            << "Number Of Positive Elements,Negative Sample Size,Zipf Parameter,Negative Universe Size,Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
               "FPR,Known FPR,Unknown FPR,Construction Time,Lookup Time,Filter Time,Disk Time\n";
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
                                              equal_fprs, use_hdd, allow_caching,
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
                                              equal_fprs, use_hdd, allow_caching,
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