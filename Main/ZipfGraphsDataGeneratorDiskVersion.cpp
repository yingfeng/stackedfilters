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
#include "../Headers/BloomFilter.h"
#include "../Headers/StackedAMQ.h"
#include "../Headers/ZipfDistribution.h"

static const int EMPIRICAL_CDF_SIZE = 100000;

std::vector<IntElement> generate_ints(uint64 num_elements) {
    std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
    std::vector<IntElement> int_vec(num_elements);
    for (uint64 i = 0; i < num_elements; i++) {
        int_vec[i] = i;
    }
    std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
    return int_vec;
}

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

double *get_collection_estimate(const std::vector<double> &pmf) {
    auto *empirical_cdf = new double[EMPIRICAL_CDF_SIZE];
    int batch_size = pmf.size() / EMPIRICAL_CDF_SIZE;
    double cumulative_probability = 0;
    for (int emp_cdf_index = 0; emp_cdf_index < EMPIRICAL_CDF_SIZE; emp_cdf_index++) {
        empirical_cdf[emp_cdf_index] = cumulative_probability;
        for (int batch_index = 0; batch_index < batch_size; batch_index++) {
            cumulative_probability += pmf[emp_cdf_index * batch_size + batch_index];
        }
    }
    return empirical_cdf;
}

struct SizeParameters {
    size_t negative_universe_size;
    size_t total_size;
    size_t num_positive;
    uint32 num_layers;
};

static double SizeFunctionVaried(unsigned num_variables, const double *rank_and_layer_fprs,
                                 double *grad, void *size_params_ptr) {
    auto *params = (SizeParameters *) size_params_ptr;
    size_t total_size = params->total_size * .995;
    size_t num_positive = params->num_positive;
    size_t num_negative = rank_and_layer_fprs[0] * params->negative_universe_size;
    uint32 num_layers = params->num_layers;
    const double *layer_fprs = rank_and_layer_fprs + 1;
    double size = 0;
    double positive_fpr = 1;
    double negative_fpr = 1;
    for (unsigned int i = 0; i < num_layers; i++) {
        if ((i % 2) == 0) {
            size += BloomFilter<IntElement>::SizeFunction(
                    layer_fprs[i], positive_fpr * num_positive);
            negative_fpr *= layer_fprs[i] * (1 + i / 5);
        } else {
            size += BloomFilter<IntElement>::SizeFunction(
                    layer_fprs[i], negative_fpr * num_negative);
            positive_fpr *= layer_fprs[i] * (1 + i / 5);
        }
    }
    return size - total_size;
}

double rank_to_psi(const double *empirical_cdf, const double rank) {
    int cdf_index = (int) (rank * EMPIRICAL_CDF_SIZE);
    if (cdf_index >= EMPIRICAL_CDF_SIZE) {
        return 1;
    }
    if (cdf_index < 0) {
        return 0;
    }
    return empirical_cdf[cdf_index];
}


struct FprParameters {
    size_t negative_universe_size;
    double penalty_coef;
    double empirical_cdf[EMPIRICAL_CDF_SIZE];
    double target_fpr;
    uint32 num_layers;
};

static double FprFunctionVaried(unsigned num_variables, const double *rank_and_layer_fprs,
                                double *grad, void *fpr_params_ptr) {
    auto *params = (FprParameters *) fpr_params_ptr;
    const double penalty_coef = params->penalty_coef;
    const double target_fpr = params->target_fpr;
    uint32 num_layers = params->num_layers;
    double psi = rank_to_psi(params->empirical_cdf, rank_and_layer_fprs[0]);
    const double *layer_fprs = rank_and_layer_fprs + 1;
    double known_fpr = layer_fprs[0];
    for (uint32 i = 1; i <= (num_layers - 1) / 2; i++) {
        known_fpr = known_fpr * layer_fprs[2 * i];
    }
    double unknown_fpr_side = 0;
    for (uint32 i = 1; i <= (num_layers - 1) / 2; i++) {
        double temp_fpr = layer_fprs[0];
        for (int j = 1; j <= 2 * (i - 1); j++) {
            temp_fpr = temp_fpr * layer_fprs[j];
        }
        unknown_fpr_side += temp_fpr * (1 - layer_fprs[2 * i - 1]);
    }
    double unknown_fpr_end = layer_fprs[0];
    for (uint32 i = 1; i <= (num_layers - 1) / 2; i++) {
        unknown_fpr_end = unknown_fpr_end * layer_fprs[i];
    }
    // Penalty Function For Number of Hashes
    int num_hashes = 0;
    for (uint32 i = 0; i < num_layers; i++) num_hashes += std::max((int) (round(-log(layer_fprs[i]) / log(2))), 1);
    return (psi * known_fpr +
            (1 - psi) * (unknown_fpr_side + unknown_fpr_end) - target_fpr) *
           (1 + num_hashes * penalty_coef);
}


std::vector<double>
CalculateLayerFPRsGivenEmpiricalCdf(const uint32 num_layers, const size_t total_size, const double fpr,
                                    const size_t num_positive_elements, const uint64 negative_universe_size,
                                    const uint64 max_known_negatives,
                                    SizeParameters &size_params, FprParameters &fpr_params) {
    uint32 num_variables = num_layers + 1;
    std::vector<double> rank_and_layer_fprs(num_variables, 0.0001);
    // Calculate the FPR of a standard one-layer filter.
    double one_level_fpr = exp(-static_cast<double>(total_size) / num_positive_elements * log(2) * log(2));
    std::vector<double> rank_and_one_layer_fprs =
            std::vector<double>(num_variables, one_level_fpr);
    for (uint32 i = 2; i < num_variables; i++) rank_and_one_layer_fprs[i] = .9999;
    if (num_layers == 1) {
        rank_and_layer_fprs = rank_and_one_layer_fprs;
        rank_and_layer_fprs[0] = 0;
        return rank_and_layer_fprs;
    }
    // Calculate the FPR of an equal-fpr stacked filter.
    auto *lower_bounds = (double *) calloc(num_variables, sizeof(double));
    for (uint32 i = 0; i < num_variables; i++) lower_bounds[i] = 0.00000000000000000001;
    auto *upper_bounds = (double *) calloc((num_variables), sizeof(double));
    for (uint32 i = 0; i < num_variables; i++) upper_bounds[i] = 1;
    double max_rank = std::min<double>((double) max_known_negatives / negative_universe_size, 1);
    upper_bounds[0] = max_rank;
    // Polish whichever layer-fpr setup has a lower fpr.
    rank_and_layer_fprs = std::vector<double>(num_variables, .5);
    rank_and_layer_fprs[0] = max_rank / 2;
    nlopt_opt local_fpr_opt = nlopt_create(NLOPT_GN_ISRES, num_variables);
    nlopt_set_lower_bounds(local_fpr_opt, lower_bounds);
    nlopt_set_upper_bounds(local_fpr_opt, upper_bounds);
    nlopt_add_inequality_constraint(local_fpr_opt, SizeFunctionVaried, &size_params, total_size * .0005);
    nlopt_set_min_objective(local_fpr_opt, FprFunctionVaried, &fpr_params);
    nlopt_set_maxtime(local_fpr_opt, .25);
    double variable_fpr_fpr = 1;
    nlopt_result local_ret_status =
            nlopt_optimize(local_fpr_opt, rank_and_layer_fprs.data(), &variable_fpr_fpr);
    if (local_ret_status == -4)
        printf("ERROR!!!  Roundoff Errors Reached in Local Optimization\n");
    else if (local_ret_status < 0)
        printf("ERROR!!! General Error in Local Optimization: %d\n", local_ret_status);
    free(upper_bounds);
    free(lower_bounds);
    return rank_and_layer_fprs;
}


void GenerateDataForOneRun(std::ofstream &file_stream, const double zipf, uint64 negative_universe_size,
                           const double bits_per_positive_element, const double fpr,
                           const bool equal_fprs, const bool use_hdd, const bool allow_caching,
                           uint32 num_layers, const uint32 num_reps, const uint64 max_known_negatives,
                           uint64 num_positives, uint64 sample_size, const double positive_rate,
                           const std::vector<IntElement> &positives, const std::vector<IntElement> &negatives,
                           double *empirical_cdf, const std::vector<double> &pmf) {

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
    const bool fixed_fpr = fpr > 0;
    uint64 num_positive_elements = positives.size();
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, negative_universe_size=%ld, bits = %f, equal_fprs=%d, num_layers=%d\n",
           positives.size(), negative_universe_size, bits_per_positive_element, equal_fprs, num_layers);
    FprParameters fpr_params;
    fpr_params.negative_universe_size = negative_universe_size;
    fpr_params.penalty_coef = .00000001;
    for (int i = 0; i < EMPIRICAL_CDF_SIZE; i++) fpr_params.empirical_cdf[i] = empirical_cdf[i];
    fpr_params.num_layers = num_layers;
    SizeParameters size_params;
    size_params.negative_universe_size = negative_universe_size;
    size_params.num_positive = num_positive_elements;
    size_params.total_size = total_size;
    size_params.num_layers = num_layers;
    if (fixed_fpr) {
        fpr_params.target_fpr = fpr;
    } else {
        fpr_params.target_fpr = 0;
    }
    std::vector<double> rank_and_layer_fprs(num_layers + 1);
    uint32 max_layers = 7;
    uint32 min_layers = 1;
    if (num_layers != 0) {
        max_layers = num_layers;
        min_layers = num_layers;
    }
    double current_minimum_fpr = 1;
    double current_minimum_size = 100000000000000;
    for (uint32 n = min_layers; n <= max_layers; n += 2) {
        fpr_params.num_layers = n;
        size_params.num_layers = n;
        for (uint32 i = 0; i < 5; i++) {
            std::vector<double> temp_rank_and_lfprs = CalculateLayerFPRsGivenEmpiricalCdf(n,
                                                                                          total_size, fpr,
                                                                                          positives.size(),
                                                                                          negative_universe_size,
                                                                                          max_known_negatives,
                                                                                          size_params,
                                                                                          fpr_params);
            if (fixed_fpr) {
                double temp_fpr = FprFunctionVaried(n + 1, temp_rank_and_lfprs.data(), nullptr,
                                                    &fpr_params);
                if (temp_fpr < current_minimum_fpr) {
                    num_layers = n;
                    rank_and_layer_fprs = temp_rank_and_lfprs;
                    current_minimum_fpr = temp_fpr;
                }
            } else {
                double temp_size = SizeFunctionVaried(n + 1, temp_rank_and_lfprs.data(), nullptr,
                                                      &size_params);
                if (temp_size < current_minimum_size) {
                    num_layers = n;
                    rank_and_layer_fprs = temp_rank_and_lfprs;
                    current_minimum_size = temp_size;
                }
            }
        }
    }

    for (uint32 reps = 0; reps < num_reps; reps++) {
        uint64 num_known_negative_elements = rank_and_layer_fprs[0] * negative_universe_size;
        number_of_chosen_negatives += num_known_negative_elements;
        std::vector<double> layer_fprs(rank_and_layer_fprs.begin() + 1, rank_and_layer_fprs.end());

        // Generate Test Data
        std::vector<IntElement> known_negatives = std::vector<IntElement>(
                negatives.begin(),
                negatives.begin() + num_known_negative_elements);
        auto construction_start = std::chrono::system_clock::now();

        // Build the Stacked Filter with the test data and layer fprs.
        StackedAMQ<BloomFilter, IntElement> filter(layer_fprs, positives, known_negatives);
        auto construction_end = std::chrono::system_clock::now();
        const auto construction_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                construction_end - construction_start);
        construction_time += construction_rep_time.count() / 1000000.0;
        constexpr double kOptTime = .1;
        construction_time += kOptTime;

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
        std::minstd_rand gen(std::random_device{}());
        std::discrete_distribution<uint64_t> dist(pmf.begin(), pmf.end());
        std::vector<IntElement> elements_to_test;
        for (uint64 i = 0; i < sample_size; i++) {
            uint64_t element_rank = dist(gen);
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
                file_pos = (sum_int +
                            elements_to_test[i].value) * 10000 % file_size;
            }
            fseek(fileptr, file_pos, SEEK_SET);
            fread(&temp_int, sizeof(uint64_t), 1, fileptr);
            sum_int += temp_int;
        }
        auto disk_end = std::chrono::high_resolution_clock::now();
        disk_time += std::chrono::duration_cast<std::chrono::microseconds>(
                disk_end - disk_start).count() / 1000000.0;


        for (uint64 i = 0; i < std::min<uint64>(num_known_negative_elements, sample_size); i++) {
            known_false_positives += filter.LookupElement(known_negatives[i]);
        }
        for (uint64 i = 0; i < sample_size; i++) {
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


    file_stream << zipf << "," << num_positive_elements << "," <<
                sample_size << "," << positive_rate << "," << negative_universe_size << ","
                << max_known_negatives << ","
                << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << equal_fprs << ","
                << num_layers << "," << used_bits << "," << total_fpr << ","
                << known_fpr << "," << unknown_fpr << "," << construction_time
                << ","
                << lookup_time << "," << filter_time << "," << disk_time << "," << rank_and_layer_fprs[1] << "\n";
    printf(", fpr=%f, pos_checks=%f, neg_checks=%f, construction_time=%f, lookup_time=%f, filter_time=%f, disk_time=%f, EFPB=%f",
           total_fpr,
           checks_per_pos, checks_per_neg, construction_time, lookup_time, filter_time, disk_time,
           rank_and_layer_fprs[1]);
    printf(" Used Bits= %f\n", used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, double &fpr_begin,
               double &fpr_min, double &fpr_ratio, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &num_positives,
               uint64 &num_positives_max, double &num_positives_ratio, uint64 &max_known_negatives,
               uint64 &max_known_negatives_max, double &max_known_negatives_ratio,
               double &positive_rate, double &positive_rate_max, double &positive_rate_step,
               bool &equal_fprs, bool &use_hdd, bool &allow_caching, uint32 &num_layers,
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
    TCLAP::ValueArg<double> positive_rate_arg("", "positive_rate",
                                              "The maximum number of known negatives available to the stacked filter.",
                                              false, 0, "Expects a double.");
    TCLAP::ValueArg<double> positive_rate_max_arg("", "positive_rate__max",
                                                  "The max value for the max number of negatives it can use.",
                                                  false, 1, "Expects a double.");
    TCLAP::ValueArg<double> positive_rate_step_arg("", "positive_rate_step",
                                                   "The size of steps over which [positive_rate, positive_rate__max) will be explored.",
                                                   false, 1.1, "Expects a double.");
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
    cmdLine.add(num_positives_arg);
    cmdLine.add(num_positives_max_arg);
    cmdLine.add(num_positives_ratio_arg);
    cmdLine.add(max_known_negatives_arg);
    cmdLine.add(max_known_negatives_max_arg);
    cmdLine.add(max_known_negatives_ratio_arg);
    cmdLine.add(positive_rate_arg);
    cmdLine.add(positive_rate_max_arg);
    cmdLine.add(positive_rate_step_arg);
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
    num_positives = num_positives_arg.getValue();
    num_positives_max = num_positives_max_arg.getValue();
    num_positives_ratio = num_positives_ratio_arg.getValue();
    max_known_negatives = max_known_negatives_arg.getValue();
    max_known_negatives_max = max_known_negatives_max_arg.getValue();
    max_known_negatives_ratio = max_known_negatives_ratio_arg.getValue();
    positive_rate = positive_rate_arg.getValue();
    positive_rate_max = positive_rate_max_arg.getValue();
    positive_rate_step = positive_rate_step_arg.getValue();
    equal_fprs = equal_fprs_arg.getValue();
    use_hdd = use_hdd_arg.getValue();
    allow_caching = allow_caching_arg.getValue();
    num_layers = num_layers_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio, neg_universe_size_ratio, max_known_negatives_ratio,
            positive_rate_begin, positive_rate_max, positive_rate_step, num_positives_ratio;
    uint64 num_positives_begin, num_positives_max, neg_universe_size_begin, neg_universe_size_max, max_known_negatives_begin, max_known_negatives_max;
    uint32 num_reps, num_layers;
    std::string file_path;
    bool equal_fprs, use_hdd, allow_caching;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, num_positives_begin, num_positives_max, num_positives_ratio,
              max_known_negatives_begin, max_known_negatives_max, max_known_negatives_ratio,
              positive_rate_begin, positive_rate_max, positive_rate_step,
              equal_fprs, use_hdd, allow_caching, num_layers, num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Zipf Parameter,Number Of Positive Elements,Sample Size,Positive Rate,Negative Universe Size,Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
               "FPR,Known FPR,Unknown FPR,Construction Time,Lookup Time,Filter Time,Disk Time,EFPB\n";
    const uint64 negative_sample_size = 10000000;

    for (double zipf = zipf_begin; zipf < zipf_max; zipf += zipf_step) {
        for (double positive_rate = positive_rate_begin;
             positive_rate < positive_rate_max; positive_rate += positive_rate_step) {
            for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
                for (double num_negatives = neg_universe_size_begin;
                     num_negatives <= neg_universe_size_max; num_negatives *= neg_universe_size_ratio) {
                    const std::vector<double> pmf = get_pmf(zipf, num_negatives);
                    double *empirical_cdf = get_collection_estimate(pmf);
                    for (double num_positives = num_positives_begin;
                         num_positives <= num_positives_max; num_positives *= num_positives_ratio) {
                        std::vector<IntElement> ints = generate_ints(num_positives + num_negatives);
                        std::vector<IntElement> positives(ints.begin(), ints.begin() + num_positives);
                        std::vector<IntElement> negatives(ints.begin() + num_positives, ints.end());
                        for (uint64 max_known_negatives = max_known_negatives_begin;
                             max_known_negatives <=
                             max_known_negatives_max; max_known_negatives *= max_known_negatives_ratio) {
                            if (fpr_begin < 0) {
                                for (double bits_per_positive_element = bits_begin;
                                     bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                                    GenerateDataForOneRun(file_stream,
                                                          zipf, num_negatives,
                                                          bits_per_positive_element,
                                                          fpr_begin,
                                                          equal_fprs, use_hdd, allow_caching,
                                                          num_layers, num_reps, max_known_negatives, num_positives,
                                                          negative_sample_size, positive_rate,
                                                          positives, negatives,
                                                          empirical_cdf, pmf);
                                    file_stream.flush();
                                }
                            } else {
                                for (double fpr = fpr_begin;
                                     fpr >= fpr_min; fpr *= fpr_ratio) {
                                    GenerateDataForOneRun(file_stream,
                                                          zipf, num_negatives,
                                                          bits_begin,
                                                          fpr,
                                                          equal_fprs, use_hdd, allow_caching,
                                                          num_layers, num_reps, max_known_negatives, num_positives,
                                                          negative_sample_size, positive_rate, positives, negatives,
                                                          empirical_cdf, pmf);
                                    file_stream.flush();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop