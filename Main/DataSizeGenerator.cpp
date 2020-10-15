#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"

#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <chrono>
#include <tclap/CmdLine.h>
#include "../Headers/BloomFilter.h"
#include "../Headers/StackedAMQ.h"
#include "../Headers/ZipfDistribution.h"

void generate_ints(std::vector<BigIntElement> &positives, std::vector<BigIntElement> &known_negatives,
                   std::vector<BigIntElement> &unknown_negatives, uint64 num_positives, uint64 num_known_negatives,
                   uint64 num_unknown_negatives) {
    for (uint64 i = 0; i < num_positives; i++) {
        positives.emplace_back(i);
    }
    for (uint64 i = 0; i < num_known_negatives; i++) {
        known_negatives.emplace_back(num_positives + i);
    }
    for (uint64 i = 0; i < num_unknown_negatives; i++) {
        unknown_negatives.emplace_back(num_positives + num_known_negatives + i);
    }
}


struct SizeParameters {
    size_t negative_elements;
    size_t total_size;
    size_t num_positive;
    uint32 num_layers;
};

static double SizeFunctionVaried(unsigned num_variables, const double *layer_fprs,
                                 double *grad, void *size_params_ptr) {
    auto *params = (SizeParameters *) size_params_ptr;
    size_t total_size = params->total_size * .995;
    size_t num_positive = params->num_positive;
    size_t num_negative = params->negative_elements;
    uint32 num_layers = params->num_layers;
    double size = 0;
    double positive_fpr = 1;
    double negative_fpr = 1;
    for (unsigned int i = 0; i < num_layers; i++) {
        if ((i % 2) == 0) {
            size += BloomFilter<IntElement>::SizeFunction(
                    layer_fprs[i], positive_fpr * num_positive);
            negative_fpr *= layer_fprs[i];
        } else {
            size += BloomFilter<IntElement>::SizeFunction(
                    layer_fprs[i], negative_fpr * num_negative);
            positive_fpr *= layer_fprs[i];
        }
    }
    return size - total_size;
}

static double SizeFunctionEqual(unsigned num_variables, const double *layer_fprs,
                                double *grad, void *size_params_ptr) {
    auto *params = (SizeParameters *) size_params_ptr;
    size_t total_size = params->total_size * .995;
    size_t num_positive = params->num_positive;
    uint32 num_layers = params->num_layers;
    size_t num_negative = params->negative_elements;
    const double layer_fpr = *layer_fprs;
    double size = 0;
    double positive_fpr = 1;
    double negative_fpr = 1;
    for (unsigned int i = 0; i < num_layers; i++) {
        if ((i % 2) == 0) {
            size += BloomFilter<IntElement>::SizeFunction(
                    layer_fpr, positive_fpr * num_positive);
            negative_fpr *= layer_fpr;
        } else {
            size += BloomFilter<IntElement>::SizeFunction(
                    layer_fpr, negative_fpr * num_negative);
            positive_fpr *= layer_fpr;
        }
    }
    return size - total_size;
}

struct FprParameters {
    double psi;
    double penalty_coef;
    double target_fpr;
    uint32 num_layers;
};

static double FprFunctionVaried(unsigned num_variables, const double *layer_fprs,
                                double *grad, void *fpr_params_ptr) {
    auto *params = (FprParameters *) fpr_params_ptr;
    const double penalty_coef = params->penalty_coef;
    const double target_fpr = params->target_fpr;
    uint32 num_layers = params->num_layers;
    double psi = params->psi;
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
    int num_hashes = 0;
    for (uint32 i = 0; i < num_layers; i++) num_hashes += std::max((int) (round(-log(layer_fprs[i]) / log(2))), 1);
    return (psi * known_fpr +
            (1 - psi) * (unknown_fpr_side + unknown_fpr_end) - target_fpr) *
           (1 + num_hashes * penalty_coef);

}

static double FprFunctionEqual(unsigned num_variables, const double *layer_fprs,
                               double *grad, void *fpr_params_ptr) {
    auto *params = (FprParameters *) fpr_params_ptr;
    const double penalty_coef = params->penalty_coef;
    const double target_fpr = params->target_fpr;
    uint32 num_layers = params->num_layers;
    double psi = params->psi;
    const double layer_fpr = *layer_fprs;
    double known_fpr = layer_fpr;
    for (uint32 i = 1; i <= (num_layers - 1) / 2; i++) {
        known_fpr = known_fpr * layer_fpr;
    }
    double unknown_fpr_side = 0;
    for (uint32 i = 1; i <= (num_layers - 1) / 2; i++) {
        double temp_fpr = layer_fpr;
        for (int j = 1; j <= 2 * (i - 1); j++) {
            temp_fpr = temp_fpr * layer_fpr;
        }
        unknown_fpr_side += temp_fpr * (1 - layer_fpr);
    }
    double unknown_fpr_end = layer_fpr;
    for (uint32 i = 1; i < num_layers; i++) {
        unknown_fpr_end = unknown_fpr_end * layer_fpr;
    }
    uint32 num_hashes = num_layers * std::max<uint32>(round(-log(layer_fpr) / log(2)), 1);
    return (psi * known_fpr +
            (1 - psi) * (unknown_fpr_side + unknown_fpr_end) - target_fpr) *
           (1 + num_hashes * penalty_coef);
}

std::vector<double>
CalculateLayerFPRsGivenZipfDistribution(const uint32 num_layers, const size_t total_size, const double fpr,
                                        const size_t num_positive_elements, const uint64 num_known_negatives,
                                        const double psi) {
    std::vector<double> layer_fprs(num_layers, 0.05);
    // Calculate the FPR of a standard one-layer filter.
    double one_level_fpr = exp(-static_cast<double>(total_size) / num_positive_elements * log(2) * log(2));
    std::vector<double> one_layer_fprs =
            std::vector<double>(num_layers, one_level_fpr);
    for (uint32 i = 2; i < num_layers; i++) one_layer_fprs[i] = .9999;
    if (num_layers == 1) {
        layer_fprs = one_layer_fprs;
        return layer_fprs;
    }
    bool fixed_fpr = (fpr > 0);

    // Put together the necessary parameter structs
    SizeParameters size_params;
    size_params.negative_elements = num_known_negatives;
    size_params.num_positive = num_positive_elements;
    size_params.total_size = total_size;
    size_params.num_layers = num_layers;
    FprParameters fpr_params;
    fpr_params.penalty_coef = .0000001;
    fpr_params.psi = psi;
    fpr_params.num_layers = num_layers;
    if (fixed_fpr) {
        fpr_params.target_fpr = fpr;
    } else {
        fpr_params.target_fpr = 0;
    }
    // Calculate the FPR of an equal-fpr stacked filter.
    auto *lower_bounds = (double *) calloc(num_layers, sizeof(double));
    for (uint32 i = 0; i < num_layers; i++) lower_bounds[i] = 0.00000000000000000001;
    auto *upper_bounds = (double *) calloc((num_layers), sizeof(double));
    for (uint32 i = 0; i < num_layers; i++) upper_bounds[i] = 1;
    nlopt_opt equal_fpr_opt = nlopt_create(NLOPT_GN_ISRES, 1);
    nlopt_set_lower_bounds(equal_fpr_opt, lower_bounds);
    nlopt_set_upper_bounds(equal_fpr_opt, upper_bounds);
    nlopt_set_maxtime(equal_fpr_opt, .5);
    if (fixed_fpr) {
        nlopt_add_inequality_constraint(
                equal_fpr_opt, FprFunctionEqual,
                &fpr_params, fpr * .01);
        nlopt_set_min_objective(equal_fpr_opt, SizeFunctionEqual,
                                &size_params);
    } else {
        nlopt_add_inequality_constraint(
                equal_fpr_opt, SizeFunctionEqual,
                &size_params, total_size * .0005);
        nlopt_set_min_objective(equal_fpr_opt, FprFunctionEqual,
                                &fpr_params);
    }
    double equal_fpr_score = 0;
    std::vector<double> single_fpr(1, .005);
    const auto equal_opt_status = nlopt_optimize(equal_fpr_opt, single_fpr.data(), &equal_fpr_score);
    if (equal_opt_status < 0)
        printf("Equal Opt Error!!  %d\n", equal_opt_status);
    uint32 num_positive_layers = (num_layers + 1) / 2;
    double equal_fpr = pow(single_fpr[0], num_positive_layers);
    // Polish whichever layer-fpr setup has a lower fpr.
    if (false) {
        printf("Using One Level Start\n");
        layer_fprs = single_fpr;
    } else {
        printf("Using Equal Level Start\n");
        layer_fprs = std::vector<double>(num_layers, single_fpr[0]);
    }
    nlopt_opt local_fpr_opt = nlopt_create(NLOPT_LN_COBYLA, num_layers);
    nlopt_set_lower_bounds(local_fpr_opt, lower_bounds);
    nlopt_set_upper_bounds(local_fpr_opt, upper_bounds);
    nlopt_set_maxtime(local_fpr_opt, .125 * pow(2, num_layers));
    if (fixed_fpr) {
        nlopt_add_inequality_constraint(
                local_fpr_opt, FprFunctionEqual,
                &fpr_params, fpr * .01);
        nlopt_set_min_objective(local_fpr_opt, SizeFunctionEqual,
                                &size_params);
    } else {
        nlopt_add_inequality_constraint(
                local_fpr_opt, SizeFunctionEqual,
                &size_params, total_size * .0005);
        nlopt_set_min_objective(local_fpr_opt, FprFunctionEqual,
                                &fpr_params);
    }
    double variable_fpr_fpr = 1;
    nlopt_result local_ret_status =
            nlopt_optimize(local_fpr_opt, layer_fprs.data(), &variable_fpr_fpr);
    if (local_ret_status == -4)
        printf("ERROR!!!  Roundoff Errors Reached in Local Optimization\n");
    else if (local_ret_status < 0)
        printf("ERROR!!! General Error in Local Optimization: %d\n", local_ret_status);
    free(upper_bounds);
    free(lower_bounds);
    return layer_fprs;
}


void GenerateDataForOneRun(std::ofstream &file_stream, const double zipf_parameter,
                           const double bits_per_positive_element, const double fpr,
                           const bool equal_fprs,
                           uint32 num_layers, const uint32 num_reps, const uint64 num_known_negatives,
                           const uint64 num_positive_elements, const double psi,
                           const uint64 negative_sample_size) {
    double known_fpr = 0;
    double unknown_fpr = 0;
    double total_fpr = 0;
    double used_bits = 0;
    double checks_per_pos = 0;
    double checks_per_neg = 0;
    double opt_time = 0;
    double construction_time = 0;
    double positive_lookup_time = 0;
    double negative_lookup_time = 0;
    uint64 number_of_chosen_negatives = 0;
    const bool fixed_fpr = fpr > 0;
    std::vector<BigIntElement> positives;
    std::vector<BigIntElement> known_negatives;
    std::vector<BigIntElement> unknown_negatives;
    generate_ints(positives, known_negatives, unknown_negatives,
                  num_positive_elements, num_known_negatives, negative_sample_size);
    // Generate Test Data
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, zipf_parameter =%f, negative_universe_size=%ld, bits = %f, equal_fprs=%d, num_layers=%d\n",
           positives.size(), zipf_parameter, 0, bits_per_positive_element, equal_fprs, num_layers);
    std::vector<double> layer_fprs;
    uint32 max_layers = 7;
    uint32 min_layers = 1;
    if (num_layers != 0) {
        max_layers = num_layers;
        min_layers = num_layers;
    }
    double current_minimum_fpr = 1;
    double current_minimum_size = 100000000000000;

    auto opt_start = std::chrono::system_clock::now();
    for (uint32 n = min_layers; n <= max_layers; n += 2) {
        for (uint32 i = 0; i < 1; i++) {
            FprParameters fpr_params;
            fpr_params.penalty_coef = .0000001;
            fpr_params.psi = psi;
            fpr_params.num_layers = n;
            SizeParameters size_params;
            size_params.negative_elements = num_known_negatives;
            size_params.num_positive = num_positive_elements;
            size_params.total_size = total_size;
            size_params.num_layers = n;
            std::vector<double> temp_layer_fprs = CalculateLayerFPRsGivenZipfDistribution(n,
                                                                                          total_size, fpr,
                                                                                          positives.size(),
                                                                                          num_known_negatives,
                                                                                          psi);
            if (fixed_fpr) {
                double temp_fpr = FprFunctionVaried(n + 1, temp_layer_fprs.data(), nullptr,
                                                    &fpr_params);
                if (temp_fpr < current_minimum_fpr) {
                    num_layers = n;
                    layer_fprs = temp_layer_fprs;
                    current_minimum_fpr = temp_fpr;
                }
            } else {
                double temp_size = SizeFunctionVaried(n + 1, temp_layer_fprs.data(), nullptr,
                                                      &size_params);
                if (temp_size < current_minimum_size) {
                    num_layers = n;
                    layer_fprs = temp_layer_fprs;
                    current_minimum_size = temp_size;
                }
            }
        }

    }
    auto opt_end = std::chrono::system_clock::now();
    opt_time = std::chrono::duration_cast<std::chrono::microseconds>(
            opt_end - opt_start).count() / 1000000.0;

    for (uint32 reps = 0; reps < num_reps; reps++) {
        number_of_chosen_negatives += num_known_negatives;
        // Build the Stacked Filter with the test data and layer fprs.
        auto construction_start = std::chrono::system_clock::now();
        StackedAMQ<BloomFilter, BigIntElement> filter(layer_fprs, positives, known_negatives);
        auto construction_end = std::chrono::system_clock::now();

        const auto construction_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                construction_end - construction_start);
        construction_time += construction_rep_time.count() / 1000000.0;

        used_bits += filter.GetSize() / (double) num_positive_elements;
        filter.PrintLayerDiagnostics();
        filter.ResetNumFilterChecks();

        checks_per_pos += (double) filter.NumFilterChecks() / num_positive_elements;
        filter.ResetNumFilterChecks();
        // Test the false positive rate for known negatives.
        uint64 false_positives = 0;
        uint64 known_false_positives = 0;
        uint64 unknown_false_positives = 0;
        uint64 known_negatives_tested = 0;
        uint64 unknown_negatives_tested = 0;
        known_negatives_tested = std::min<uint64>(num_known_negatives, negative_sample_size);
        unknown_negatives_tested = negative_sample_size;
        auto negative_lookup_start = std::chrono::system_clock::now();
        for (uint64 i = 0; i < known_negatives_tested; i++) {
            known_false_positives += filter.LookupElement(known_negatives[i]);
        }
        for (uint64 i = 0; i < unknown_negatives_tested; i++) {
            unknown_false_positives += filter.LookupElement(unknown_negatives[i]);
        }
        auto negative_lookup_end = std::chrono::system_clock::now();

        const auto negative_lookup_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                negative_lookup_end - negative_lookup_start);
        negative_lookup_time += negative_lookup_rep_time.count() / 1000000.0;

        checks_per_neg +=
                (double) filter.NumFilterChecks() / negative_sample_size;
        filter.ResetNumFilterChecks();

        double trial_known_fpr = (double) (known_false_positives) / (double) (known_negatives_tested);
        double trial_unknown_fpr =
                (double) (unknown_false_positives) / (double) (unknown_negatives_tested);
        known_fpr += trial_known_fpr;
        unknown_fpr += trial_unknown_fpr;
        total_fpr += psi * trial_known_fpr + (1 - psi) * trial_unknown_fpr;
        printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f\n",
               psi * trial_known_fpr + (1 - psi) * trial_unknown_fpr,
               num_known_negatives, psi);
    }
    known_fpr /= num_reps;
    unknown_fpr /= num_reps;
    total_fpr /= num_reps;
    used_bits /= num_reps;
    checks_per_neg /= num_reps;
    checks_per_pos /= num_reps;
    number_of_chosen_negatives /= num_reps;
    construction_time /= num_reps;
    negative_lookup_time /= num_reps;
    positive_lookup_time /= num_reps;

    file_stream << num_positive_elements << "," <<
                negative_sample_size << "," << zipf_parameter << "," << 0 << ","
                << num_known_negatives << ","
                << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << equal_fprs << ","
                << num_layers << "," << used_bits << "," << total_fpr << ","
                << known_fpr << "," << unknown_fpr << "," << construction_time << "," << positive_lookup_time
                << ","
                << negative_lookup_time << "," << opt_time << "," << checks_per_pos << ","
                << checks_per_neg << "," << layer_fprs[1] << "\n";
    printf(", fpr=%f, pos_checks=%f, neg_checks=%f, construction_time=%f, optimization_time=%f, positive_lookup_time=%f, negative_lookup_time=%f, EFPB=%f",
           total_fpr,
           checks_per_pos, checks_per_neg, construction_time, opt_time, positive_lookup_time, negative_lookup_time,
           layer_fprs[1]);
    printf(" Used Bits= %f\n", used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, double &fpr_begin,
               double &fpr_min, double &fpr_ratio, uint64 &num_positives_begin, uint64 &num_positives_max,
               double &num_positives_ratio, double &known_negatives_ratio,
               double &known_negatives_ratio_max, double &known_negatives_ratio_step, double &psi, double &psi_max,
               double &psi_step, bool &equal_fprs, uint32 &num_layers,
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
    TCLAP::ValueArg<uint64> num_positives_arg("", "num_positives",
                                              "The starting value for the negative universe size, and the only value if neg_universe_size_max is not set.",
                                              false, 1000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> num_positives_max_arg("", "num_positives_max",
                                                  "The max value for the number of known negatives.",
                                                  false, 100000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> num_positives_ratio_arg("", "num_positives_ratio",
                                                    "The ratio between steps over which [known_negatives_ratio, known_negatives_ratio_max) will be explored.",
                                                    false, 1000000000, "Expects a double.");
    TCLAP::ValueArg<double> known_negatives_ratio_arg("", "known_negatives_ratio",
                                                      "The starting value for the negative universe size, and the only value if neg_universe_size_max is not set.",
                                                      false, 1, "Expects a uint64.");
    TCLAP::ValueArg<double> known_negatives_ratio_max_arg("", "known_negatives_ratio_max",
                                                          "The max value for the number of known negatives.",
                                                          false, 10000, "Expects a uint64.");
    TCLAP::ValueArg<double> known_negatives_ratio_step_arg("", "known_negatives_ratio_step",
                                                           "The ratio between steps over which [known_negatives_ratio, known_negatives_ratio_max) will be explored.",
                                                           false, 100000000, "Expects a double.");
    TCLAP::ValueArg<double> psi_arg("", "psi",
                                    "The probability that a negative query lands in the known negative set.",
                                    false, .5, "Expects a double.");
    TCLAP::ValueArg<double> psi_max_arg("", "psi_max",
                                        "The max probability that a negative query lands in the known negative set.",
                                        false, 1, "Expects a double.");
    TCLAP::ValueArg<double> psi_step_arg("", "psi_step",
                                         "The step size over which [psi, psi_max] will be explored.",
                                         false, 1, "Expects a double.");
    TCLAP::ValueArg<bool> equal_fprs_arg("", "equal_fprs",
                                         "Whether all layer fprs should be made equal.",
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
    cmdLine.add(num_positives_arg);
    cmdLine.add(num_positives_max_arg);
    cmdLine.add(num_positives_ratio_arg);
    cmdLine.add(known_negatives_ratio_arg);
    cmdLine.add(known_negatives_ratio_max_arg);
    cmdLine.add(known_negatives_ratio_step_arg);
    cmdLine.add(psi_arg);
    cmdLine.add(psi_max_arg);
    cmdLine.add(psi_step_arg);
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
    num_positives_begin = num_positives_arg.getValue();
    num_positives_max = num_positives_max_arg.getValue();
    num_positives_ratio = num_positives_ratio_arg.getValue();
    known_negatives_ratio = known_negatives_ratio_arg.getValue();
    known_negatives_ratio_max = known_negatives_ratio_max_arg.getValue();
    known_negatives_ratio_step = known_negatives_ratio_step_arg.getValue();
    psi = psi_arg.getValue();
    psi_max = psi_max_arg.getValue();
    psi_step = psi_step_arg.getValue();
    equal_fprs = equal_fprs_arg.getValue();
    num_layers = num_layers_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio, num_positives_ratio,
            psi_min, psi_max, psi_step, known_negatives_ratio_begin, known_negatives_ratio_max, known_negatives_ratio_step;
    uint64 num_positives_begin, num_positives_max;
    uint32 num_reps, num_layers;
    std::string file_path;
    bool equal_fprs;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, fpr_begin, fpr_min, fpr_ratio,
              num_positives_begin, num_positives_max, num_positives_ratio,
              known_negatives_ratio_begin, known_negatives_ratio_max, known_negatives_ratio_step, psi_min, psi_max,
              psi_step,
              equal_fprs, num_layers,
              num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Number Of Positive Elements,Negative Sample Size,Zipf Parameter,Negative Universe Size,Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
               "FPR,Known FPR,Unknown FPR,Construction Time,Positive Lookup Time,Negative Lookup Time,Optimization Time,Filter Checks For Positive,Filter Checks For "
               "Negative,EFPB\n";
    const uint64 negative_sample_size = 10000000;
    for (uint64 num_positives = num_positives_begin;
         num_positives < num_positives_max; num_positives *= num_positives_ratio) {
        for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
            if (zipf_parameter <= 1.01 && zipf_parameter >= .99) continue;
            for (double psi = psi_min;
                 psi <= psi_max; psi += psi_step) {
                for (double known_negatives_ratio = known_negatives_ratio_begin;
                     known_negatives_ratio <=
                     known_negatives_ratio_max; known_negatives_ratio += known_negatives_ratio_step) {
                    if (fpr_begin < 0) {
                        for (double bits_per_positive_element = bits_begin;
                             bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                            GenerateDataForOneRun(file_stream, zipf_parameter,
                                                  bits_per_positive_element,
                                                  fpr_begin,
                                                  equal_fprs,
                                                  num_layers, num_reps, num_positives * known_negatives_ratio,
                                                  num_positives, psi,
                                                  negative_sample_size);
                            file_stream.flush();
                        }
                    } else {
                        for (double fpr = fpr_begin;
                             fpr >= fpr_min; fpr *= fpr_ratio) {
                            GenerateDataForOneRun(file_stream, zipf_parameter,
                                                  bits_begin,
                                                  fpr,
                                                  equal_fprs,
                                                  num_layers, num_reps, num_positives * known_negatives_ratio,
                                                  num_positives, psi,
                                                  negative_sample_size);
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