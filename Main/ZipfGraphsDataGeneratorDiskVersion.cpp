#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
//
// Created by kylebd99 on 8/30/19.
//

#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <unistd.h>
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

void GenerateDataForOneRun(std::ofstream &file_stream, const double zipf, uint64 negative_universe_size,
                           const double bits_per_positive_element, const bool use_hdd, const bool allow_caching,
                           const uint32 num_reps, const uint64 max_known_negatives, uint64 sample_size,
                           const std::vector<IntElement> &positives, const std::vector<IntElement> &negatives) {

    if (use_hdd) sample_size /= 10;

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
    long double psi = 0;
    uint64 num_positive_elements = positives.size();
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, negative_universe_size=%ld, bits = %f\n",
           positives.size(), negative_universe_size, bits_per_positive_element);
    std::vector<double> cdf = zipf_cdf(negative_universe_size, zipf);
    for (uint32 reps = 0; reps < num_reps; reps++) {
        // Generate Test Data
        auto construction_start = std::chrono::system_clock::now();
        // Build the Stacked Filter with the test data and layer fprs.
        StackedFilter<BloomFilterLayer, IntElement> filter(total_size, positives, negatives, cdf);
        auto construction_end = std::chrono::system_clock::now();
        const auto construction_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                construction_end - construction_start);
        construction_time += construction_rep_time.count() / 1000000.0;

        uint64 num_known_negative_elements = filter.num_negative_;
        number_of_chosen_negatives += num_known_negative_elements;

        used_bits += filter.GetSize() / (double) num_positive_elements;
        filter.PrintLayerDiagnostics();
        filter.ResetNumFilterChecks();

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
        std::minstd_rand gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(0, 1);
        std::vector<IntElement> negatives_to_test;
        for (uint64 i = 0; i < sample_size; i++) {
            double uniform_double = dist(gen);
            uint64 element_rank = inverseCdfFast(uniform_double, sample_size, negative_universe_size);
            negatives_to_test.push_back(negatives[element_rank]);
            if (element_rank < num_known_negative_elements) {
                known_negatives_tested++;
            } else {
                unknown_negatives_tested++;
            }
        }

        // Filter Timing
        auto filter_start = std::chrono::high_resolution_clock::now();
        for (const auto &element : negatives_to_test) {
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
                file_pos = (negatives_to_test[i].value * 100000) % file_size;
            } else {
                file_pos = (sum_int +
                            negatives_to_test[i].value) * 10000 % file_size;
            }
            fseek(fileptr, file_pos, SEEK_SET);
            fread(&temp_int, sizeof(uint64_t), 1, fileptr);
            sum_int += temp_int;
        }
        auto disk_end = std::chrono::high_resolution_clock::now();
        disk_time += std::chrono::duration_cast<std::chrono::microseconds>(
                disk_end - disk_start).count() / 1000000.0;


        for (uint64 i = 0; i < std::min<uint64>(num_known_negative_elements, sample_size); i++) {
            known_false_positives += filter.LookupElement(negatives[i]);
        }
        for (uint64 i = 0;
             i < std::min<uint64>(negative_universe_size - num_known_negative_elements, sample_size); i++) {
            unknown_false_positives += filter.LookupElement(negatives[i + num_known_negative_elements]);
        }


        known_fpr += (double) (known_false_positives) /
                     (double) (std::min<uint64>(num_known_negative_elements, sample_size));
        unknown_fpr +=
                (double) (unknown_false_positives) / (double) (sample_size);
        total_fpr += (double) false_positives / (double) sample_size;
        psi += (double) known_negatives_tested / sample_size;
        printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f, sum_int:%ld\n",
               (double) false_positives / (double) sample_size,
               num_known_negative_elements, (double) known_negatives_tested / sample_size, sum_int);
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


    file_stream << zipf << "," << num_positive_elements << "," << sample_size << "," << negative_universe_size << ","
                << max_known_negatives << "," << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << used_bits << "," << total_fpr << ","
                << known_fpr << "," << unknown_fpr << "," << construction_time
                << "," << lookup_time << "," << filter_time << "," << disk_time << "\n";
    printf(", fpr=%f, pos_checks=%f, neg_checks=%f, construction_time=%f, lookup_time=%f, filter_time=%f, disk_time=%f",
           total_fpr, checks_per_pos, checks_per_neg, construction_time, lookup_time, filter_time, disk_time);
    printf(" Used Bits= %f\n", used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &num_positives,
               uint64 &num_positives_max, double &num_positives_ratio, uint64 &max_known_negatives,
               uint64 &max_known_negatives_max, double &max_known_negatives_ratio,
               bool &use_hdd, bool &allow_caching,
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
                                                  false, 10000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> neg_universe_size_max_arg("", "neg_universe_size_max",
                                                      "The max value for the negative universe size.",
                                                      false, 10000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> neg_universe_size_ratio_arg("", "neg_universe_size_ratio",
                                                        "The ratio between steps over which [neg_universe_size, neg_universe_size_max) will be explored.",
                                                        false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint64> num_positives_arg("", "num_positives",
                                              "The starting value for the number of positive elements, and the only value if num_positives_max is not set.",
                                              false, 1000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> num_positives_max_arg("", "num_positives_max",
                                                  "The max value for the number of positive elements.",
                                                  false, 10000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> num_positives_ratio_arg("", "num_positives_ratio",
                                                    "The ratio between steps over which [num_positives, num_positives_max) will be explored.",
                                                    false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint64> max_known_negatives_arg("", "max_known_negatives",
                                                    "The maximum number of known negatives available to the stacked filter.",
                                                    false, 100000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> max_known_negatives_max_arg("", "max_known_negatives_max",
                                                        "The max value for the max number of negatives it can use.",
                                                        false, 100000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> max_known_negatives_ratio_arg("", "max_known_negatives_ratio",
                                                          "The ratio between steps over which [max_known_negatives, max_known_negatives_max) will be explored.",
                                                          false, 100000000, "Expects a double.");
    TCLAP::ValueArg<bool> use_hdd_arg("", "use_hdd",
                                      "Whether hdd should be used to read.",
                                      false, false, "Expects a bool.");
    TCLAP::ValueArg<bool> allow_caching_arg("", "allow_caching",
                                            "Whether caching effects should be minimized.",
                                            false, false, "Expects a bool.");
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
    cmdLine.add(neg_universe_size_arg);
    cmdLine.add(neg_universe_size_max_arg);
    cmdLine.add(neg_universe_size_ratio_arg);
    cmdLine.add(num_positives_arg);
    cmdLine.add(num_positives_max_arg);
    cmdLine.add(num_positives_ratio_arg);
    cmdLine.add(max_known_negatives_arg);
    cmdLine.add(max_known_negatives_max_arg);
    cmdLine.add(max_known_negatives_ratio_arg);
    cmdLine.add(use_hdd_arg);
    cmdLine.add(allow_caching_arg);
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
    num_positives = num_positives_arg.getValue();
    num_positives_max = num_positives_max_arg.getValue();
    num_positives_ratio = num_positives_ratio_arg.getValue();
    max_known_negatives = max_known_negatives_arg.getValue();
    max_known_negatives_max = max_known_negatives_max_arg.getValue();
    max_known_negatives_ratio = max_known_negatives_ratio_arg.getValue();
    use_hdd = use_hdd_arg.getValue();
    allow_caching = allow_caching_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, neg_universe_size_ratio, max_known_negatives_ratio,
            positive_rate_begin, positive_rate_max, positive_rate_step, num_positives_ratio;
    uint64 num_positives_begin, num_positives_max, neg_universe_size_begin, neg_universe_size_max, max_known_negatives_begin, max_known_negatives_max;
    uint32 num_reps;
    std::string file_path;
    bool use_hdd, allow_caching;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, num_positives_begin, num_positives_max, num_positives_ratio,
              max_known_negatives_begin, max_known_negatives_max, max_known_negatives_ratio,
              use_hdd, allow_caching, num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Zipf Parameter,Number Of Positive Elements,Sample Size,Negative Universe Size,"
               "Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Used Bits,Total "
               "FPR,Known FPR,Unknown FPR,Construction Time,Lookup Time,Filter Time,Disk Time\n";
    const uint64 negative_sample_size = 10000000;

    for (double zipf = zipf_begin; zipf < zipf_max; zipf += zipf_step) {
        for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
            for (double num_negatives = neg_universe_size_begin;
                 num_negatives <= neg_universe_size_max; num_negatives *= neg_universe_size_ratio) {
                for (double num_positives = num_positives_begin;
                     num_positives <= num_positives_max; num_positives *= num_positives_ratio) {
                    std::vector<IntElement> ints = generate_ints(num_positives + num_negatives);
                    std::vector<IntElement> positives(ints.begin(), ints.begin() + num_positives);
                    std::vector<IntElement> negatives(ints.begin() + num_positives, ints.end());
                    for (uint64 max_known_negatives = max_known_negatives_begin;
                         max_known_negatives <=
                         max_known_negatives_max; max_known_negatives *= max_known_negatives_ratio) {
                        for (double bits_per_positive_element = bits_begin;
                             bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                            GenerateDataForOneRun(file_stream, zipf, num_negatives, bits_per_positive_element, use_hdd,
                                                  allow_caching, num_reps, max_known_negatives, negative_sample_size,
                                                  positives, negatives);
                            file_stream.flush();
                        }
                    }
                }

            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop