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
#include "powerlaw.h"


static const int EMPIRICAL_CDF_SIZE = 10000;
static const int NUM_POSITIVES = 10000;
static const int NUM_NEGATIVES = 90000;


std::vector<IntElement> get_positives() {
    std::vector<IntElement> positives;
    for (int i = 1; i <= NUM_POSITIVES; i++) {
        positives.emplace_back(i);
    }
    return positives;
}

std::vector<IntElement> get_negatives() {
    std::vector<IntElement> negatives;
    for (int i = NUM_POSITIVES; i <= NUM_POSITIVES + NUM_NEGATIVES; i++) {
        negatives.emplace_back(i);
    }
    return negatives;
}


std::pair<std::vector<double>, std::vector<double>> get_cdf_pmf() {
    unsigned int max_random = 1000000;
    power_law_distribution_t *distribution = power_law_initialize(.5, NUM_NEGATIVES, max_random);
    std::vector<double> cdf(NUM_NEGATIVES);
    for (int element_idx = 0; element_idx < NUM_NEGATIVES; element_idx++) {
        cdf[element_idx] = static_cast<double>(distribution->key_cdf[element_idx]) / max_random;
    }
    std::vector<double> pmf(NUM_NEGATIVES);
    pmf[0] = cdf[0];
    for (int element_idx = 1; element_idx < NUM_NEGATIVES; element_idx++) {
        pmf[element_idx] = cdf[element_idx] - cdf[element_idx - 1];
    }
    return {cdf, pmf};
}

void GenerateDataForOneRun(std::ofstream &file_stream, uint64 negative_universe_size,
                           const double bits_per_positive_element, const bool use_hdd, const bool allow_caching,
                           const uint32 num_reps, const uint64 max_known_negatives, uint64 sample_size,
                           const std::vector<IntElement> &positives, const std::vector<IntElement> &negatives,
                           const std::vector<double> &cdf, const std::vector<double> &pmf) {
    double total_fpr = 0;
    double used_bits = 0;
    uint64 number_of_chosen_negatives = 0;
    long double psi = 0;
    uint64 num_positive_elements = positives.size();
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, negative_universe_size=%ld, bits = %f\n",
           positives.size(), negative_universe_size, bits_per_positive_element);
    // Reduce the number of samples for HDD experiments
    if (use_hdd) {
        sample_size /= 10;
    }
    for (uint32 reps = 0; reps < num_reps; reps++) {
        // Limit the negatives to the max number allowed
        std::vector<IntElement> potential_known_negatives = std::vector<IntElement>(
                negatives.begin(),
                negatives.begin() + max_known_negatives);
        std::vector<double> potential_known_negatives_cdf = std::vector<double>(
                cdf.begin(),
                cdf.begin() + max_known_negatives);
        StackedFilter<BloomFilterLayer, IntElement> filter(bits_per_positive_element * positives.size(), positives,
                                                           potential_known_negatives, potential_known_negatives_cdf);
        used_bits += filter.GetSize() / (double) num_positive_elements;
        filter.PrintLayerDiagnostics();
        uint64 num_known_negative_elements = filter.num_negative_;

        // Test the false positive rate for known negatives.
        uint64 false_positives = 0;
        uint64 known_negatives_tested = 0;

        // Generate the sample queries.
        std::minstd_rand gen(std::random_device{}());
        std::discrete_distribution<uint64_t> dist(pmf.begin(), pmf.end());
        std::vector<IntElement> elements_to_test;
        for (uint64 i = 0; i < sample_size; i++) {
            uint64_t element_rank = dist(gen);
            elements_to_test.push_back(negatives[element_rank]);
            if (element_rank < num_known_negative_elements) {
                known_negatives_tested++;
            }
        }

        // Filter Timing
        for (const auto &element : elements_to_test) {
            const bool is_false_positive = filter.LookupElement(element);
            if (is_false_positive) {
                false_positives++;
            }
        }

        total_fpr += (double) false_positives / (double) sample_size;
        psi += (double) known_negatives_tested / sample_size;
        printf("Trial FPR:%f, Negative Elements:%ld\n",
               (double) false_positives / (double) sample_size, num_known_negative_elements);
    }
    total_fpr /= num_reps;
    used_bits /= num_reps;
    number_of_chosen_negatives /= num_reps;
    psi /= num_reps;

    file_stream << num_positive_elements << "," <<
                sample_size << "," << negative_universe_size << ","
                << max_known_negatives << ","
                << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << used_bits << "," << total_fpr << "\n";
    printf(", FPR=%f, Used Bits= %f\n",
           total_fpr, used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_known_negatives,
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
                                                  false, NUM_NEGATIVES, "Expects a uint64.");
    TCLAP::ValueArg<uint64> neg_universe_size_max_arg("", "neg_universe_size_max",
                                                      "The max value for the negative universe size.",
                                                      false, NUM_NEGATIVES, "Expects a uint64.");
    TCLAP::ValueArg<double> neg_universe_size_ratio_arg("", "neg_universe_size_ratio",
                                                        "The ratio between steps over which [neg_universe_size, neg_universe_size_max) will be explored.",
                                                        false, 2, "Expects a double.");
    TCLAP::ValueArg<uint64> max_known_negatives_arg("", "max_known_negatives",
                                                    "The maximum number of known negatives available to the stacked filter.",
                                                    false, 10000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> max_known_negatives_max_arg("", "max_known_negatives_max",
                                                        "The max value for the max number of negatives it can use.",
                                                        false, 100000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> max_known_negatives_ratio_arg("", "max_known_negatives_ratio",
                                                          "The ratio between steps over which [max_known_negatives, max_known_negatives_max) will be explored.",
                                                          false, 10000000000, "Expects a double.");
    TCLAP::ValueArg<bool> use_hdd_arg("", "use_hdd",
                                      "Whether hdd should be used to read.",
                                      false, false, "Expects a bool.");
    TCLAP::ValueArg<bool> allow_caching_arg("", "allow_caching",
                                            "Whether caching effects should be minimized.",
                                            false, false, "Expects a bool.");
    TCLAP::ValueArg<uint32> num_reps_arg("", "num_reps",
                                         "The number of repetitions per set of parameters.",
                                         false, 100, "Expects a uint32.");
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
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, neg_universe_size_ratio, max_known_negatives_ratio;
    uint64 neg_universe_size_begin, neg_universe_size_max, max_known_negatives_begin, max_known_negatives_max;
    uint32 num_reps;
    std::string file_path;
    bool use_hdd, allow_caching;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, max_known_negatives_begin, max_known_negatives_max, max_known_negatives_ratio,
              use_hdd, allow_caching,
              num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Number Of Positive Elements,Sample Size,Negative Universe Size,"
               "Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Used Bits,Total "
               "FPR,Construction Time,Lookup Time,Filter Time,Disk Time\n";
    const uint64 negative_sample_size = 1000000;
    std::vector<IntElement> positives = get_positives();
    std::vector<double> cdf, pmf;
    std::tie(cdf, pmf) = get_cdf_pmf();

    for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
        if (zipf_parameter <= 1.01 && zipf_parameter >= .99) continue;
        for (double neg_universe_size = neg_universe_size_begin;
             neg_universe_size <= neg_universe_size_max; neg_universe_size *= neg_universe_size_ratio) {
            std::vector<IntElement> negatives = get_negatives();
            for (uint64 max_known_negatives = max_known_negatives_begin;
                 max_known_negatives <=
                 max_known_negatives_max; max_known_negatives *= max_known_negatives_ratio) {
                for (double bits_per_positive_element = bits_begin;
                     bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                    GenerateDataForOneRun(file_stream,
                                          neg_universe_size,
                                          bits_per_positive_element, use_hdd, allow_caching,
                                          num_reps, max_known_negatives, negative_sample_size,
                                          positives, negatives, cdf, pmf);
                    file_stream.flush();
                }
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop