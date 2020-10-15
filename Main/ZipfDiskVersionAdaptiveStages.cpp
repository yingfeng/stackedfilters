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
#include <experimental/algorithm>
#include <iterator>
#include "../Headers/BloomFilter.h"
#include "../Headers/AdaptiveStackedBF.h"
#include "../Headers/ZipfDistribution.h"

#define POSITIVE_ELEMENTS 1000000
#define ZIPF_PARAMETER_1 1.25
#define ZIPF_PARAMETER_2 1

std::vector<IntElement> generate_ints(uint64 num_elements) {
    std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
    std::vector<IntElement> int_vec(num_elements);
    for (uint64 i = 0; i < num_elements; i++) {
        int_vec[i] = i;
    }
    std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
    return int_vec;
}

std::vector<double> get_pmf(const double zipf_parameter, const uint64_t negative_universe_size) {
    std::vector<double> pmf(negative_universe_size);
    double summation = 0;
    for (uint64_t idx = 1; idx < negative_universe_size + 1; idx++) {
        pmf[idx] = 1 / pow(idx, zipf_parameter);
        summation += pmf[idx];
    }
    for (auto &p : pmf) p /= summation;
    return pmf;
}


uint64_t ReadDisk(FILE *fileptr, uint64_t file_pos) {
    uint64_t temp_int;
    fseek(fileptr, file_pos, SEEK_SET);
    fread(&temp_int, sizeof(uint64_t), 1, fileptr);
    return temp_int;
}

// To reduce caching effects, a similarly sized chunk of data is read from the random file
// rather than reading the positives file over and over.
size_t simulate_reading_positives(char *file_name, size_t num_positives, size_t iteration) {
    FILE *fileptr = fopen(file_name, "r");
    fseek(fileptr, 0L, SEEK_END);
    uint64_t file_size = ftell(fileptr);
    uint64_t positives_bytes = sizeof(uint64_t) * num_positives;
    size_t chunks = trunc((double) file_size / positives_bytes);
    size_t cur_chunk = iteration % chunks;
    fseek(fileptr, cur_chunk * positives_bytes, SEEK_SET);
    uint64_t sum_int = 0;
    uint64_t temp_int = 0;
    for (size_t pos_idx = 0; pos_idx < num_positives; pos_idx++) {
        fread(&temp_int, sizeof(uint64_t), 1, fileptr);
        sum_int += temp_int % 123456789;
    }
    return sum_int;
}

IntElement GetQuery(std::discrete_distribution<uint64_t> &dist, std::minstd_rand &gen,
                    const std::vector<IntElement> &negatives) {
    return negatives[dist(gen)];
}

void MarkTimestamp(std::vector<double> &timestamps, std::vector<uint64_t> &queries,
                   std::chrono::time_point<std::chrono::system_clock> start, uint64_t query_idx) {
    auto timestamp = std::chrono::system_clock::now();
    auto timestamp_dif = std::chrono::duration_cast<std::chrono::microseconds>(
            timestamp - start).count() / 1000000.0;
    timestamps.push_back(timestamp_dif);
    queries.push_back(query_idx);
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-msc30-c"

void GenerateDataForOneRun(std::ofstream &file_stream, const uint64 negative_universe_size,
                           const double bits_per_positive_element,
                           const bool use_hdd,
                           const uint32 num_reps, uint64 max_sample_size, uint64 total_queries,
                           const std::vector<IntElement> &positives,
                           const std::vector<IntElement> &negatives, size_t sample_estimate_size,
                           const std::vector<double> &pmf_1,
                           const std::vector<double> &pmf_2,
                           const std::vector<double> &pmf_3) {

    constexpr uint64_t kQueriesPerTimestamp = 10000;
    const uint64_t num_timestamps = total_queries / kQueriesPerTimestamp + 1;

    // Setup Tracking Variables
    size_t number_of_chosen_negatives = 0;
    size_t sample_queries = 0;
    double used_bits_adaptive = 0;
    std::vector<double> adaptive_timestamps(num_timestamps, 0);
    std::vector<uint64_t> adaptive_queries_completed(num_timestamps, 0);

    double used_bits_traditional = 0;
    std::vector<double> traditional_timestamps(num_timestamps, 0);
    std::vector<uint64_t> traditional_queries_completed(num_timestamps, 0);

    // Setup Misc Variables
    std::minstd_rand gen(std::random_device{}());
    std::discrete_distribution<uint64_t> dist_1(pmf_1.begin(), pmf_1.end());
    std::discrete_distribution<uint64_t> dist_2(pmf_2.begin(), pmf_2.end());
    std::discrete_distribution<uint64_t> dist_3(pmf_3.begin(), pmf_3.end());
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
    uint64_t sum_int = 0;
    uint64 num_positive_elements = positives.size();
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, negative_universe_size=%ld, max_sample_size=%ld, bits = %f\n",
           positives.size(), negative_universe_size, max_sample_size, bits_per_positive_element);

    for (uint32 reps = 0; reps < num_reps; reps++) {

        std::vector<double> adaptive_timestamps_rep;
        std::vector<uint64_t> adaptive_queries_completed_rep;
        std::vector<double> traditional_timestamps_rep;
        std::vector<uint64_t> traditional_queries_completed_rep;



        // Build the query workload from three stages.
        uint64_t stage_length = total_queries / 3;
        std::vector<IntElement> queries(total_queries, 0);
        for (size_t query_idx = 0; query_idx < stage_length; query_idx++)
            queries[query_idx] = GetQuery(dist_1, gen, negatives);
        for (size_t query_idx = stage_length; query_idx < 2 * stage_length; query_idx++)
            queries[query_idx] = GetQuery(dist_2, gen, negatives);
        for (size_t query_idx = 2 * stage_length; query_idx < 3 * stage_length; query_idx++)
            queries[query_idx] = GetQuery(dist_3, gen, negatives);



        // Construct the adaptive filter then run it and collect timestamps throughout the workload.
        auto adaptive_start = std::chrono::system_clock::now();
        AdaptiveStackedBF<IntElement> adaptive_filter(positives, total_size, total_queries, pmf_1,
                                                      sample_estimate_size);

        for (uint64_t query_idx = 0; query_idx < total_queries; query_idx++) {
            if ((query_idx % kQueriesPerTimestamp) == 0) {
                MarkTimestamp(adaptive_timestamps_rep, adaptive_queries_completed_rep, adaptive_start, query_idx);
            }
            if (adaptive_filter.LookupElement(queries[query_idx])) {
                file_pos = (sum_int +
                            negatives[(query_idx * query_idx + sum_int) % negatives.size()].value) * 10000 % file_size;
                sum_int = (sum_int + ReadDisk(fileptr, file_pos)) % 123456789;
                auto status = adaptive_filter.DeclareFalsePositiveAndCheckStatus(queries[query_idx]);
                if (status == NEEDS_THIRD_LAYER) {
                    printf("Building Third Layer\n");
                    sum_int += simulate_reading_positives(file_name, num_positive_elements, rand());
                    adaptive_filter.BuildThirdLayer(positives);
                } else if (status == NEEDS_REBUILD) {
                    printf("Rebuilding: Without Re-Optimization\n");
                    sum_int += simulate_reading_positives(file_name, num_positive_elements, rand());
                    adaptive_filter.RebuildFilter();
                }
            }
        }
        MarkTimestamp(adaptive_timestamps_rep, adaptive_queries_completed_rep, adaptive_start, total_queries);
        used_bits_adaptive += adaptive_filter.GetSize() / static_cast<double>(num_positive_elements);

        // Construct the traditional filter then run it and collect timestamps throughout the workload.
        auto traditional_start = std::chrono::system_clock::now();
        sum_int += simulate_reading_positives(file_name, num_positive_elements, rand());
        uint32_t traditional_num_hashes =
                (double) total_size / static_cast<double>(positives.size()) * (double) logf64(2.);
        BloomFilter<IntElement> traditional_filter(total_size, traditional_num_hashes, rand());
        for (const auto &element: positives) traditional_filter.InsertElement(element);
        MarkTimestamp(traditional_timestamps_rep, traditional_queries_completed_rep, traditional_start, 0);
        for (uint64_t query_idx = 0; query_idx < total_queries; query_idx++) {
            if ((query_idx % kQueriesPerTimestamp) == 0) {
                MarkTimestamp(traditional_timestamps_rep, traditional_queries_completed_rep, traditional_start,
                              query_idx);
            }
            if (traditional_filter.LookupElement(queries[query_idx])) {
                file_pos = (sum_int +
                            negatives[(query_idx * query_idx + sum_int) % negatives.size()].value) * 10000 % file_size;
                sum_int = (sum_int + ReadDisk(fileptr, file_pos)) % 123456789;
            }
        }
        MarkTimestamp(traditional_timestamps_rep, traditional_queries_completed_rep, traditional_start, total_queries);
        used_bits_traditional += traditional_filter.GetSize() / static_cast<double>(num_positive_elements);

        for (uint64_t timestamp_idx = 0; timestamp_idx < num_timestamps; timestamp_idx++) {
            adaptive_timestamps[timestamp_idx] +=
                    static_cast<double>(adaptive_timestamps_rep[timestamp_idx]) / num_reps;
            adaptive_queries_completed[timestamp_idx] +=
                    static_cast<double>(adaptive_queries_completed_rep[timestamp_idx]) / num_reps;
            traditional_timestamps[timestamp_idx] +=
                    static_cast<double>(traditional_timestamps_rep[timestamp_idx]) / num_reps;
            traditional_queries_completed[timestamp_idx] +=
                    static_cast<double>(traditional_queries_completed_rep[timestamp_idx]) / num_reps;
        }
        printf("False Positives Collected:%ld, sum_int:%ld\n", adaptive_filter.false_positives_capacity_, sum_int);
    }

    number_of_chosen_negatives /= num_reps;
    sample_queries /= num_reps;
    used_bits_adaptive /= num_reps;
    used_bits_traditional /= num_reps;
    for (uint64_t timestamp_idx = 0; timestamp_idx < num_timestamps; timestamp_idx++) {
        file_stream << num_positive_elements << "," << total_queries << "," << negative_universe_size << ","
                    << max_sample_size << "," << number_of_chosen_negatives << "," << sample_queries << ","
                    << bits_per_positive_element << "," << used_bits_adaptive << "," << used_bits_traditional << ","
                    << adaptive_timestamps[timestamp_idx] << "," << adaptive_queries_completed[timestamp_idx] << ","
                    << traditional_timestamps[timestamp_idx] << "," << traditional_queries_completed[timestamp_idx]
                    << "\n";
    }
    printf(", used bits= %f\n", used_bits_adaptive);
    printf(", trad used bits= %f\n", used_bits_traditional);
    printf("Adaptive Total Time:%f, Trad Total Time:%f\n",
           adaptive_timestamps[adaptive_timestamps.size() - 1],
           traditional_timestamps[traditional_timestamps.size() - 1]);
}

#pragma clang diagnostic pop

void set_flags(TCLAP::CmdLine &cmdLine, double &bits_begin,
               double &bits_max, double &bits_step, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_sample_size,
               uint64 &max_sample_size_max, double &max_sample_size_ratio, uint64 &total_queries, bool &use_hdd,
               uint32 &num_reps, std::string &file_path, int arg_num, char **args) {

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
    TCLAP::ValueArg<uint64> max_sample_size_arg("", "max_sample_size",
                                                "The maximum number of known negatives available to the stacked filter.",
                                                false, 100000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> max_sample_size_max_arg("", "max_sample_size_max",
                                                    "The max value for the max number of negatives it can use.",
                                                    false, 100000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> max_sample_size_ratio_arg("", "max_sample_size_ratio",
                                                      "The ratio between steps over which [max_known_negatives, max_known_negatives_max) will be explored.",
                                                      false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint64> total_queries_arg("", "total_queries",
                                              "The number of queries processed.",
                                              false, 10000000, "Expects a uint64.");
    TCLAP::ValueArg<bool> use_hdd_arg("", "use_hdd",
                                      "Whether hdd should be used to read.",
                                      false, false, "Expects a bool.");
    TCLAP::ValueArg<uint32> num_reps_arg("", "num_reps",
                                         "The number of repetitions per set of parameters.",
                                         false, 25, "Expects a uint32.");
    TCLAP::ValueArg<std::string> file_path_arg("", "path",
                                               "The path to the output file, relative or absolute.",
                                               false, "Data/TestData.csv", "Expects a string.");
    cmdLine.add(bits_begin_arg);
    cmdLine.add(bits_max_arg);
    cmdLine.add(bits_step_arg);
    cmdLine.add(neg_universe_size_arg);
    cmdLine.add(neg_universe_size_max_arg);
    cmdLine.add(neg_universe_size_ratio_arg);
    cmdLine.add(max_sample_size_arg);
    cmdLine.add(max_sample_size_max_arg);
    cmdLine.add(max_sample_size_ratio_arg);
    cmdLine.add(total_queries_arg);
    cmdLine.add(use_hdd_arg);
    cmdLine.add(num_reps_arg);
    cmdLine.add(file_path_arg);
    cmdLine.parse(arg_num, args);

    bits_begin = bits_begin_arg.getValue();
    bits_max = bits_max_arg.getValue();
    bits_step = bits_step_arg.getValue();
    neg_universe_size = neg_universe_size_arg.getValue();
    neg_universe_size_max = neg_universe_size_max_arg.getValue();
    neg_universe_size_ratio = neg_universe_size_ratio_arg.getValue();
    max_sample_size = max_sample_size_arg.getValue();
    max_sample_size_max = max_sample_size_max_arg.getValue();
    max_sample_size_ratio = max_sample_size_ratio_arg.getValue();
    total_queries = total_queries_arg.getValue();
    use_hdd = use_hdd_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double bits_begin, bits_max, bits_step, neg_universe_size_ratio, max_sample_size_ratio;
    uint64 neg_universe_size_begin, neg_universe_size_max, max_sample_size_begin, max_sample_size_max, total_queries;
    uint32 num_reps;
    std::string file_path;
    bool use_hdd;
    set_flags(cmd, bits_begin, bits_max, bits_step,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, max_sample_size_begin, max_sample_size_max,
              max_sample_size_ratio, total_queries, use_hdd,
              num_reps, file_path,
              arg_num, args
    );

    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Positives Size,Total Queries,Negative Universe Size,Max Sample Size,False Positives Chosen,"
               "Sample Queries, Bits Available,Used Bits Adaptive,Used Bits Traditional,"
               "Adaptive Timestamps,Adaptive Queries,Traditional Timestamps,Traditional Queries\n";
    size_t sample_estimate_size = 25;
    for (double neg_universe_size = neg_universe_size_begin;
         neg_universe_size < neg_universe_size_max; neg_universe_size *= neg_universe_size_ratio) {
        std::vector<IntElement> ints = generate_ints(POSITIVE_ELEMENTS + neg_universe_size);
        std::vector<IntElement> positives(ints.begin(), ints.begin() + POSITIVE_ELEMENTS);
        std::vector<IntElement> negatives(ints.begin() + POSITIVE_ELEMENTS, ints.end());
        const std::vector<double> pmf_1 = get_pmf(ZIPF_PARAMETER_1, neg_universe_size);
        std::vector<double> pmf_2(neg_universe_size, 0);
        std::reverse_copy(std::begin(pmf_1), std::end(pmf_1), std::begin(pmf_2));
        const std::vector<double> pmf_3_temp = get_pmf(ZIPF_PARAMETER_2, neg_universe_size);
        std::vector<double> pmf_3(neg_universe_size, 0);
        std::reverse_copy(std::begin(pmf_3_temp), std::end(pmf_3_temp), std::begin(pmf_3));
        for (uint64 max_sample_size = max_sample_size_begin;
             max_sample_size <= max_sample_size_max; max_sample_size *= max_sample_size_ratio) {
            if (max_sample_size >= total_queries) max_sample_size = total_queries;
            for (double bits_per_positive_element = bits_begin;
                 bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                GenerateDataForOneRun(file_stream, neg_universe_size, bits_per_positive_element, use_hdd, num_reps,
                                      max_sample_size, total_queries, positives, negatives, sample_estimate_size,
                                      pmf_1, pmf_2, pmf_3);
                file_stream.flush();
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop