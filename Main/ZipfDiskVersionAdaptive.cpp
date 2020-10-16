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
#define ZIPF_PARAMETER .75

std::vector<IntElement> generate_ints(uint64 num_elements) {
    std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
    std::vector<IntElement> int_vec(num_elements);
    for (uint64 i = 0; i < num_elements; i++) {
        int_vec[i] = i;
    }
    std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
    return int_vec;
}

std::vector<double> get_cdf(const double zipf_parameter, const uint64_t negative_universe_size) {
    std::vector<double> cdf(negative_universe_size);
    double summation = 0;
    for (uint64_t idx = 1; idx < negative_universe_size + 1; idx++) {
        summation += 1 / pow(idx, zipf_parameter);
        cdf[idx] = summation;
    }
    for (auto &p : cdf) p /= summation;
    return cdf;
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

void GenerateDataForOneRun(std::ofstream &file_stream, const uint64 negative_universe_size,
                           const double bits_per_positive_element,
                           const bool use_hdd,
                           const uint32 num_reps, uint64 max_sample_size, uint64 total_queries,
                           const std::vector<IntElement> &positives,
                           const std::vector<IntElement> &negatives, size_t sample_estimate_size,
                           const std::vector<double> &pmf,
                           const std::vector<double> &cdf) {

    // Setup Tracking Variables
    size_t number_of_chosen_negatives = 0;
    size_t sample_queries = 0;
    double used_bits_adaptive = 0;
    double cold_construction_adaptive = 0;
    double warm_construction_adaptive = 0;
    double cold_lookup_adaptive = 0;
    double warm_lookup_adaptive = 0;
    double cold_disk_adaptive = 0;
    double warm_disk_adaptive = 0;
    double cold_fpr_adaptive = 0;
    double warm_fpr_adaptive = 0;

    double used_bits_traditional = 0;
    double construction_traditional = 0;
    double lookup_traditional = 0;
    double disk_traditional = 0;
    double fpr_traditional = 0;

    // Setup Misc Variables
    std::minstd_rand gen(std::random_device{}());
    std::discrete_distribution<uint64_t> dist(pmf.begin(), pmf.end());
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
        // Cold Construction Adaptive: build the Stacked Filter with the positives and sample cdf estimates.
        auto cold_construction_adaptive_start = std::chrono::system_clock::now();
        sum_int += simulate_reading_positives(file_name, num_positive_elements, rand());
        AdaptiveStackedBF<IntElement> adaptive_filter(positives, total_size, total_queries, cdf);
        auto cold_construction_adaptive_end = std::chrono::system_clock::now();
        cold_construction_adaptive += std::chrono::duration_cast<std::chrono::microseconds>(
                cold_construction_adaptive_end - cold_construction_adaptive_start).count() / 1000000.0;
        adaptive_filter.PrintLayerDiagnostics();

        std::vector<IntElement> queries(total_queries);
        for (size_t query_idx = 0; query_idx < total_queries; query_idx++) {
            queries[query_idx] = GetQuery(dist, gen, negatives);
        }

        // Cold Run Adaptive: Run the Stacked Filter and flag false positives until it fills, then perform disk reads.
        size_t queries_until_full = 0;
        size_t cold_false_positives_adaptive = 0;
        size_t query_idx = 0;
        auto cold_run_lookup_adaptive_start = std::chrono::system_clock::now();
        for (query_idx = 0; query_idx < total_queries; query_idx++) {
            if (adaptive_filter.LookupElement(queries[query_idx])) {
                cold_false_positives_adaptive++;
                FilterStatus status = adaptive_filter.DeclareFalsePositiveAndCheckStatus(queries[query_idx]);
                if (status != OPERATIONAL) {
                    break;
                }
            }
        }
        queries_until_full = query_idx;
        auto cold_run_lookup_adaptive_end = std::chrono::system_clock::now();
        cold_lookup_adaptive += std::chrono::duration_cast<std::chrono::microseconds>(
                cold_run_lookup_adaptive_end - cold_run_lookup_adaptive_start).count() / 1000000.0;
        cold_fpr_adaptive += static_cast<double>(cold_false_positives_adaptive) / queries_until_full;
        sample_queries += queries_until_full;

        auto cold_run_disk_adaptive_start = std::chrono::high_resolution_clock::now();
        for (uint64 i = 0; i < cold_false_positives_adaptive; i++) {
            file_pos = (sum_int +
                        negatives[(i * i + sum_int) % negatives.size()].value) * 10000 % file_size;
            sum_int = (sum_int + ReadDisk(fileptr, file_pos)) % 123456789;
        }
        auto cold_run_disk_adaptive_end = std::chrono::high_resolution_clock::now();
        cold_disk_adaptive += std::chrono::duration_cast<std::chrono::microseconds>(
                cold_run_disk_adaptive_end - cold_run_disk_adaptive_start).count() / 1000000.0;

        // Warm Construction Adaptive: Build the third layer of the Stacked Filter.
        auto warm_construction_adaptive_start = std::chrono::system_clock::now();
        sum_int += simulate_reading_positives(file_name, num_positive_elements, rand());
        adaptive_filter.BuildThirdLayer(positives);
        auto warm_construction_adaptive_end = std::chrono::system_clock::now();
        warm_construction_adaptive += std::chrono::duration_cast<std::chrono::microseconds>(
                warm_construction_adaptive_end - warm_construction_adaptive_start).count() / 1000000.0;
        adaptive_filter.PrintLayerDiagnostics();

        // Warm Run Adaptive: Run with the full filter.
        size_t warm_queries = total_queries - queries_until_full;
        size_t warm_false_positives_adaptive = 0;
        auto warm_lookup_adaptive_start = std::chrono::system_clock::now();
        for (query_idx = queries_until_full; query_idx < total_queries; query_idx++) {
            if (adaptive_filter.LookupElement(queries[query_idx])) {
                warm_false_positives_adaptive++;
            }
        }
        auto warm_lookup_adaptive_end = std::chrono::system_clock::now();
        warm_lookup_adaptive += std::chrono::duration_cast<std::chrono::microseconds>(
                warm_lookup_adaptive_end - warm_lookup_adaptive_start).count() / 1000000.0;
        warm_fpr_adaptive += static_cast<double>(warm_false_positives_adaptive) / warm_queries;

        auto warm_disk_adaptive_start = std::chrono::high_resolution_clock::now();
        for (uint64 i = 0; i < warm_false_positives_adaptive; i++) {
            file_pos = (sum_int +
                        negatives[(i * i + sum_int) % negatives.size()].value) * 10000 % file_size;
            sum_int = (sum_int + ReadDisk(fileptr, file_pos)) % 123456789;
        }
        auto warm_disk_adaptive_end = std::chrono::high_resolution_clock::now();
        warm_disk_adaptive += std::chrono::duration_cast<std::chrono::microseconds>(
                warm_disk_adaptive_end - warm_disk_adaptive_start).count() / 1000000.0;
        used_bits_adaptive += adaptive_filter.GetSize() / (double) num_positive_elements;
        number_of_chosen_negatives += adaptive_filter.false_positives_capacity_;

        // Construction Traditional: build the traditional Filter with the positives.
        auto construction_traditional_start = std::chrono::system_clock::now();
        sum_int += simulate_reading_positives(file_name, num_positive_elements, rand());
        uint32_t traditional_num_hashes =
                (double) total_size / static_cast<double>(positives.size()) * (double) logf64(2.);
        BloomFilter<IntElement> traditional_filter(total_size, traditional_num_hashes, rand());
        for (const auto &positive : positives) traditional_filter.InsertElement(positive);
        auto construction_traditional_end = std::chrono::system_clock::now();
        construction_traditional += std::chrono::duration_cast<std::chrono::microseconds>(
                construction_traditional_end - construction_traditional_start).count() / 1000000.0;
        used_bits_traditional += traditional_filter.GetSize() / (double) num_positive_elements;

        // Run Traditional: Run the Stacked Filter and flag false positives until it fills, then perform disk reads.
        size_t false_positives_traditional = 0;
        auto lookup_traditional_start = std::chrono::system_clock::now();
        for (query_idx = 0; query_idx < total_queries; query_idx++) {
            if (traditional_filter.LookupElement(queries[query_idx])) {
                false_positives_traditional++;
            }
        }
        auto lookup_traditional_end = std::chrono::system_clock::now();
        lookup_traditional += std::chrono::duration_cast<std::chrono::microseconds>(
                lookup_traditional_end - lookup_traditional_start).count() / 1000000.0;
        fpr_traditional += static_cast<double>(false_positives_traditional) / total_queries;

        auto disk_traditional_start = std::chrono::high_resolution_clock::now();
        for (uint64 i = 0; i < false_positives_traditional; i++) {
            file_pos = (sum_int +
                        negatives[(i * i + sum_int) % negatives.size()].value) * 10000 % file_size;
            sum_int = (sum_int + ReadDisk(fileptr, file_pos)) % 123456789;
        }
        auto disk_traditional_end = std::chrono::high_resolution_clock::now();
        disk_traditional += std::chrono::duration_cast<std::chrono::microseconds>(
                disk_traditional_end - disk_traditional_start).count() / 1000000.0;

        printf("Trial FPR:%f, False Positives Collected:%ld, sum_int:%ld COLD\n",
               (double) cold_false_positives_adaptive / (double) queries_until_full,
               0L, sum_int);
        printf("Trial FPR:%f, False Positives Collected:%ld, sum_int:%ld WARM\n",
               (double) warm_false_positives_adaptive / (double) warm_queries,
               adaptive_filter.false_positives_capacity_, sum_int);
    }

    used_bits_adaptive /= num_reps;
    number_of_chosen_negatives /= num_reps;
    sample_queries /= num_reps;
    cold_construction_adaptive /= num_reps;
    warm_construction_adaptive /= num_reps;
    cold_lookup_adaptive /= num_reps;
    warm_lookup_adaptive /= num_reps;
    cold_disk_adaptive /= num_reps;
    warm_disk_adaptive /= num_reps;
    cold_fpr_adaptive /= num_reps;
    warm_fpr_adaptive /= num_reps;
    used_bits_traditional /= num_reps;
    construction_traditional /= num_reps;
    lookup_traditional /= num_reps;
    disk_traditional /= num_reps;
    fpr_traditional /= num_reps;

    file_stream << num_positive_elements << "," << total_queries << "," << negative_universe_size << ","
                << max_sample_size << "," << number_of_chosen_negatives << "," << sample_queries << ","
                << bits_per_positive_element << ","
                << used_bits_adaptive << "," << cold_construction_adaptive << "," << warm_construction_adaptive << ","
                << cold_lookup_adaptive << "," << warm_lookup_adaptive << "," << cold_disk_adaptive << ","
                << warm_disk_adaptive << "," << cold_fpr_adaptive << "," << warm_fpr_adaptive << ","
                << used_bits_traditional << "," << construction_traditional << "," << lookup_traditional << ","
                << disk_traditional << "," << fpr_traditional << "\n";
    printf("cold fpr=%f, cold construction_time=%f, cold lookup_time=%f, cold disk_time=%f",
           cold_fpr_adaptive, cold_construction_adaptive, cold_lookup_adaptive, cold_disk_adaptive);
    printf(", warm fpr=%f, warm construction_time=%f, warm lookup_time=%f, warm disk_time=%f",
           warm_fpr_adaptive, warm_construction_adaptive, warm_lookup_adaptive, warm_disk_adaptive);
    printf(", used bits= %f, sample_queries=%ld\n", used_bits_adaptive, sample_queries);
    printf("trad fpr=%f, trad construction_time=%f, trad lookup_time=%f, trad disk_time=%f",
           fpr_traditional, construction_traditional, lookup_traditional, disk_traditional);
    printf(", trad used bits= %f\n", used_bits_traditional);
    printf("Adaptive Total Time:%f, Trad Total Time:%f\n",
           cold_construction_adaptive + cold_disk_adaptive + cold_lookup_adaptive + warm_construction_adaptive +
           warm_disk_adaptive + warm_lookup_adaptive, construction_traditional + disk_traditional + lookup_traditional);
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
            << "Positives Size,Total Queries,Negative Universe Size,Max Sample Size,False Positives Chosen,Sample Queries, Bits Available,Used Bits Adaptive,"
               "Cold Construction Adaptive,Warm Construction Adaptive,Cold Lookup Adaptive,Warm Lookup Adaptive,"
               "Cold Disk Adaptive,Warm Disk Adaptive,Cold FPR Adaptive,Warm FPR Adaptive,Used Bits Traditional,"
               "Construction Traditional,Lookup Traditional,Disk Traditional,FPR Traditional\n";
    size_t sample_estimate_size = 25;
    for (double neg_universe_size = neg_universe_size_begin;
         neg_universe_size < neg_universe_size_max; neg_universe_size *= neg_universe_size_ratio) {
        std::vector<IntElement> ints = generate_ints(POSITIVE_ELEMENTS + neg_universe_size);
        std::vector<IntElement> positives(ints.begin(), ints.begin() + POSITIVE_ELEMENTS);
        std::vector<IntElement> negatives(ints.begin() + POSITIVE_ELEMENTS, ints.end());
        const std::vector<double> pmf = get_pmf(ZIPF_PARAMETER, neg_universe_size);
        const std::vector<double> cdf = get_cdf(ZIPF_PARAMETER, neg_universe_size);
        for (uint64 max_sample_size = max_sample_size_begin;
             max_sample_size <= max_sample_size_max; max_sample_size *= max_sample_size_ratio) {
            if (max_sample_size >= total_queries) max_sample_size = total_queries;
            for (double bits_per_positive_element = bits_begin;
                 bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                GenerateDataForOneRun(file_stream, neg_universe_size, bits_per_positive_element, use_hdd, num_reps,
                                      max_sample_size, total_queries, positives, negatives, sample_estimate_size,
                                      pmf, cdf);
                file_stream.flush();
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop