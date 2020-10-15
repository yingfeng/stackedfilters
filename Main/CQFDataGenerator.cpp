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
#include <assert.h>
#include "../Headers/CQFilter.h"
#include "../Headers/StackedAMQ.h"
#include "../Headers/ZipfDistribution.h"

namespace {
    std::vector<IntElement> generate_ints(uint64 num_elements) {
        std::uniform_int_distribution<int> distribution(0, 0xFFFFFFFF);
        std::vector<IntElement> int_vec(num_elements);
        for (uint64 i = 0; i < num_elements; i++) {
            int_vec[i] = i;
        }
        std::shuffle(int_vec.begin(), int_vec.end(), std::minstd_rand());
        return int_vec;
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
        uint64 total_size = params->total_size * .995;
        uint64 num_positive = params->num_positive;
        uint64 num_negative = rank_and_layer_fprs[0] * params->negative_universe_size;
        uint32 num_layers = params->num_layers;
        std::vector<double> layer_fprs;
        for (int i = 1; i < num_variables; i++) {
            layer_fprs.push_back(rank_and_layer_fprs[i]);
        }
        uint64 size = 0;
        double positive_fpr = 1;
        double negative_fpr = 1;
        for (unsigned int i = 0; i < num_layers; i++) {
            if ((i % 2) == 0) {
                size += CQFilter<IntElement>::SizeFunction(
                        layer_fprs[i], positive_fpr * num_positive);
                negative_fpr *= layer_fprs[i];
            } else {
                size += CQFilter<IntElement>::SizeFunction(
                        layer_fprs[i], negative_fpr * num_negative);
                positive_fpr *= layer_fprs[i];
            }
        }
        return size - total_size;
    }

    static double SizeFunctionEqual(unsigned num_variables, const double *rank_and_layer_fprs,
                                    double *grad, void *size_params_ptr) {
        auto *params = (SizeParameters *) size_params_ptr;
        uint64 total_size = params->total_size * .995;
        uint64 num_positive = params->num_positive;
        uint32 num_layers = params->num_layers;
        uint64 num_negative = rank_and_layer_fprs[0] * params->negative_universe_size;
        const double layer_fpr = rank_and_layer_fprs[1];
        uint64 size = 0;
        double positive_fpr = 1;
        double negative_fpr = 1;
        for (unsigned int i = 0; i < num_layers; i++) {
            if ((i % 2) == 0) {
                size += CQFilter<IntElement>::SizeFunction(
                        layer_fpr, positive_fpr * num_positive);
                negative_fpr *= layer_fpr;
            } else {
                size += CQFilter<IntElement>::SizeFunction(
                        layer_fpr, negative_fpr * num_negative);
                positive_fpr *= layer_fpr;
            }
        }
        return size - total_size;
    }

    struct FprParameters {
        size_t negative_universe_size;
        double zipf_parameter;
        double penalty_coef;
        double zipf_denominator;
        uint32 num_layers;
        double target_fpr;
    };

    static double FprFunctionVaried(unsigned num_variables, const double *rank_and_layer_fprs,
                                    double *grad, void *fpr_params_ptr) {
        auto *params = (FprParameters *) fpr_params_ptr;
        const double penalty_coef = params->penalty_coef;
        const size_t negative_universe_size = params->negative_universe_size;
        const double zipf_parameter = params->zipf_parameter;
        const double zipf_denominator = params->zipf_denominator;
        uint32 num_layers = params->num_layers;
        double psi = approx_zipf_cdf(rank_and_layer_fprs[0] * negative_universe_size, zipf_denominator, zipf_parameter);
        std::vector<double> layer_fprs;
        double positive_fpr = 1;
        double negative_fpr = 1;
        for (uint32 i = 1; i < num_variables; i++) {
            // Because remainder bits can only be added integrally, this is the true fpr expression.
            if ((i % 2) == 1) {
                layer_fprs.push_back(pow(2, -std::max<int>(ceil(-log2(rank_and_layer_fprs[i])), 2)));
                negative_fpr *= layer_fprs[i];
            } else {
                layer_fprs.push_back(pow(2, -std::max<int>(ceil(-log2(rank_and_layer_fprs[i])), 2)));
                positive_fpr *= layer_fprs[i];
            }
        }
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
        return (psi * known_fpr +
                (1 - psi) * (unknown_fpr_side + unknown_fpr_end)) - params->target_fpr;

    }

    static double FprFunctionEqual(unsigned num_variables, const double *rank_and_layer_fprs,
                                   double *grad, void *fpr_params_ptr) {
        FprParameters *params = (FprParameters *) fpr_params_ptr;
        const double penalty_coef = params->penalty_coef;
        const size_t negative_universe_size = params->negative_universe_size;
        const double zipf_parameter = params->zipf_parameter;
        const double zipf_denominator = params->zipf_denominator;
        const double num_layers = params->num_layers;
        double psi = approx_zipf_cdf(rank_and_layer_fprs[0] * negative_universe_size, zipf_denominator, zipf_parameter);
        const double layer_fpr = rank_and_layer_fprs[1];
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
        return (psi * known_fpr +
                (1 - psi) * (unknown_fpr_side + unknown_fpr_end)) - params->target_fpr;
    }

    std::vector<double>
    CalculateLayerFPRsGivenZipfDistribution(const uint32 num_layers, const size_t total_size, const double fpr,
                                            const size_t num_positive_elements, const uint64 negative_universe_size,
                                            const double zipf_parameter, const uint64 max_known_negatives) {
        uint32 num_variables = num_layers + 1;
        std::vector<double> rank_and_layer_fprs(num_variables, 0.05);
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
        bool fixed_fpr = (fpr > 0);

        // Put together the necessary parameter structs
        SizeParameters size_params;
        size_params.negative_universe_size = negative_universe_size;
        size_params.num_positive = num_positive_elements;
        size_params.total_size = total_size;
        size_params.num_layers = num_layers;
        FprParameters fpr_params;
        fpr_params.negative_universe_size = negative_universe_size;
        fpr_params.penalty_coef = .0000001;
        fpr_params.zipf_parameter = zipf_parameter;
        fpr_params.zipf_denominator = approx_zipf_denominator(negative_universe_size, zipf_parameter);
        fpr_params.num_layers = num_layers;
        if (fixed_fpr) {
            fpr_params.target_fpr = fpr;
        } else {
            fpr_params.target_fpr = 0;
        }
        // Calculate the FPR of an equal-fpr stacked filter.
        auto *lower_bounds = (double *) calloc(num_variables, sizeof(double));
        for (uint32 i = 0; i < num_variables; i++) lower_bounds[i] = 0.00000000000000000001;
        auto *upper_bounds = (double *) calloc((num_variables), sizeof(double));
        for (uint32 i = 0; i < num_variables; i++) upper_bounds[i] = 1;
        double max_rank = std::min<double>((double) max_known_negatives / negative_universe_size, 1);
        upper_bounds[0] = max_rank;
        nlopt_opt equal_fpr_opt = nlopt_create(NLOPT_GN_ISRES, 2);
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
        std::vector<double> rank_and_single_fpr(2, .05);
        const auto equal_opt_status = nlopt_optimize(equal_fpr_opt, rank_and_single_fpr.data(), &equal_fpr_score);
        if (equal_opt_status < 0)
            printf("Equal Opt Error!!  %d\n", equal_opt_status);
        uint32 num_positive_layers = (num_layers + 1) / 2;
        double equal_fpr = pow(rank_and_single_fpr[1], num_positive_layers);
        // Polish whichever layer-fpr setup has a lower fpr.
        if (equal_fpr > one_level_fpr) {
            printf("Using One Level Start\n");
            rank_and_layer_fprs = rank_and_one_layer_fprs;
        } else {
            printf("Using Equal Level Start\n");
            rank_and_layer_fprs = std::vector<double>(num_variables, rank_and_single_fpr[1]);
            rank_and_layer_fprs[0] = rank_and_single_fpr[0];
        }
        nlopt_opt local_fpr_opt = nlopt_create(NLOPT_LN_COBYLA, num_variables);
        nlopt_set_lower_bounds(local_fpr_opt, lower_bounds);
        nlopt_set_upper_bounds(local_fpr_opt, upper_bounds);
        nlopt_set_maxtime(local_fpr_opt, 1 + num_layers * .025);
        nlopt_set_ftol_rel(local_fpr_opt, .0001);
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
                nlopt_optimize(local_fpr_opt, rank_and_layer_fprs.data(), &variable_fpr_fpr);
        if (local_ret_status == -4)
            printf("ERROR!!!  Roundoff Errors Reached in Local Optimization\n");
        else if (local_ret_status < 0)
            printf("ERROR!!! General Error in Local Optimization: %d\n", local_ret_status);
        free(upper_bounds);
        free(lower_bounds);
        return rank_and_layer_fprs;
    }


    std::vector<double>
    ThreeLayerCalculateOptimalFPRsGivenPsi(const uint32 num_layers, const size_t total_size, const double fpr,
                                           const size_t num_positive, const double psi, const size_t num_negative) {
        constexpr uint32 k_HI = 20;
        constexpr uint32 k_LO = 2;
        std::vector<double> layer_fprs = {1, 1, 1};
        double min_fpr = 1;
        uint64 min_size = pow(2, 60);
        bool fixed_fpr = (fpr > 0);//TODO implement FPR->Size optimization
        for (uint32 k1 = k_LO; k1 < k_HI; k1++) {
            double layer_1_fpr = 1 / pow(2, k1);
            uint64 layer_1_size = CQFilter<IntElement>::SizeFunction(layer_1_fpr, num_positive);
            if (layer_1_size >= total_size) {
                break;
            }
            for (uint32 k2 = k_LO; k2 < k_HI; k2++) {
                double layer_2_fpr = 1 / pow(2, k2);
                uint64 layer_2_size = CQFilter<IntElement>::SizeFunction(layer_2_fpr, layer_1_fpr * num_negative);
                if (layer_1_size + layer_2_size >= total_size) {
                    break;
                }
                for (uint32 k3 = k_LO; k3 < k_HI; k3++) {
                    double layer_3_fpr = 1 / pow(2, k3);
                    uint64 layer_3_size = CQFilter<IntElement>::SizeFunction(layer_3_fpr,
                                                                             layer_2_fpr * num_positive);
                    if (layer_1_size + layer_2_size + layer_3_size > total_size) {
                        break;
                    }
                    double total_fpr =
                            psi * layer_1_fpr * layer_3_fpr + (1 - psi) * (layer_1_fpr * (1 - layer_2_fpr) +
                                                                           layer_1_fpr * layer_2_fpr *
                                                                           layer_3_fpr);
                    if (min_fpr > total_fpr) {
                        layer_fprs = {layer_1_fpr, layer_2_fpr, layer_3_fpr};
                        min_fpr = total_fpr;
                    }

                }
            }
        }
        return layer_fprs;
    }

    std::vector<double>
    ThreeLayerCalculateOptimalRankAndFPRs(const uint32 num_layers, const size_t total_size, const double fpr,
                                          const size_t num_positive, const uint64 negative_universe_size,
                                          const double zipf_parameter, const uint64 max_known_negatives) {
        const double zipf_denominator = approx_zipf_denominator(negative_universe_size, zipf_parameter);
        double min_fpr = 1;
        std::vector<double> rank_and_layer_fprs(num_layers + 1, .005);
        const double max_rank = std::min<double>(static_cast<double>(max_known_negatives) / negative_universe_size, 1);
        // rank scanned over (0,1) in geometric space
        for (double rank = 0.00119519642; rank <= max_rank; rank *= 1.4) {
            const double psi = approx_zipf_cdf(static_cast<uint64>(rank * negative_universe_size), zipf_denominator,
                                               zipf_parameter);
            std::vector<double> temp_layer_fprs = ThreeLayerCalculateOptimalFPRsGivenPsi(num_layers, total_size,
                                                                                         0, num_positive, psi,
                                                                                         rank *
                                                                                         negative_universe_size);
            const double total_fpr = psi * (temp_layer_fprs[0] * temp_layer_fprs[1]) +
                                     (1 - psi) * (temp_layer_fprs[0] * (1 - temp_layer_fprs[1]) +
                                                  temp_layer_fprs[0] * temp_layer_fprs[1] * temp_layer_fprs[2]);
            if (min_fpr > total_fpr) {
                min_fpr = total_fpr;
                rank_and_layer_fprs[0] = rank;
                rank_and_layer_fprs[1] = temp_layer_fprs[0];
                rank_and_layer_fprs[2] = temp_layer_fprs[1];
                rank_and_layer_fprs[3] = temp_layer_fprs[2];
            }
        }
        // Try one iteration with no negative elements taken.
        std::vector<double> temp_layer_fprs = ThreeLayerCalculateOptimalFPRsGivenPsi(num_layers, total_size,
                                                                                     0, num_positive, 0,
                                                                                     0 *
                                                                                     negative_universe_size);
        const double total_fpr = (temp_layer_fprs[0] * (1 - temp_layer_fprs[1]) +
                                  temp_layer_fprs[0] * temp_layer_fprs[1] * temp_layer_fprs[2]);
        if (min_fpr > total_fpr) {
            min_fpr = total_fpr;
            rank_and_layer_fprs[0] = 0;
            rank_and_layer_fprs[1] = temp_layer_fprs[0];
            rank_and_layer_fprs[2] = temp_layer_fprs[1];
            rank_and_layer_fprs[3] = temp_layer_fprs[2];
        }
        if (min_fpr == 1) {
            printf("Optimizing ERROR!!!\n");
        }
        return rank_and_layer_fprs;
    }

    void
    GenerateDataForOneRun(std::ofstream &file_stream, const double zipf_parameter, const uint64 negative_universe_size,
                          const double bits_per_positive_element, const double fpr,
                          const bool equal_fprs,
                          const uint32 num_layers, const uint32 num_reps, const uint64 max_known_negatives,
                          const uint64 num_positives, const uint64 negative_sample_size) {
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
        long double psi = 0;
        const bool fixed_fpr = fpr > 0;
        std::vector<IntElement> ints = generate_ints(num_positives + negative_universe_size);
        std::vector<IntElement> positives(ints.begin(), ints.begin() + num_positives);
        uint64 num_positive_elements = positives.size();
        std::vector<IntElement> negatives(ints.begin() + num_positive_elements, ints.end());
        uint64 total_size = num_positive_elements * bits_per_positive_element;
        printf("num_positive_elements =%ld, zipf_parameter =%f, negative_universe_size=%ld, bits = %f, equal_fprs=%d, num_layers=%d\n",
               positives.size(), zipf_parameter, negative_universe_size, bits_per_positive_element, equal_fprs,
               num_layers);
        std::vector<double> rank_and_layer_fprs = {0, 0};
        if (num_layers == 1) {
            for (uint32 k1 = 2; k1 < 20; k1++) {
                double layer_1_fpr = 1 / pow(2, k1);
                uint64 layer_1_size = CQFilter<IntElement>::SizeFunction(layer_1_fpr, num_positive_elements);
                if (layer_1_size >= total_size) {
                    break;
                }
                rank_and_layer_fprs[1] = layer_1_fpr;
            }
        } else if (num_layers == 3) {
            rank_and_layer_fprs = ThreeLayerCalculateOptimalRankAndFPRs(num_layers, total_size, fpr,
                                                                        num_positive_elements,
                                                                        negative_universe_size,
                                                                        zipf_parameter,
                                                                        max_known_negatives);
        } else {
            rank_and_layer_fprs = CalculateLayerFPRsGivenZipfDistribution(num_layers, total_size, fpr,
                                                                          num_positive_elements,
                                                                          negative_universe_size,
                                                                          zipf_parameter,
                                                                          max_known_negatives);
        }
        FprParameters fpr_params;
        fpr_params.negative_universe_size = negative_universe_size;
        fpr_params.penalty_coef = .0000001;
        fpr_params.zipf_parameter = zipf_parameter;
        fpr_params.zipf_denominator = approx_zipf_denominator(negative_universe_size, zipf_parameter);
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
        double current_minimum_fpr = 1;
        double current_minimum_size = 100000000000000;
        for (uint32 i = 0; i < 5; i++) {
            std::vector<double> temp_rank_and_lfprs = rank_and_layer_fprs;
            if (num_layers == 3 || num_layers == 1) {
                break;
            } else {
                rank_and_layer_fprs = CalculateLayerFPRsGivenZipfDistribution(num_layers, total_size, fpr,
                                                                              num_positive_elements,
                                                                              negative_universe_size,
                                                                              zipf_parameter,
                                                                              max_known_negatives);
            }
            if (fixed_fpr) {
                double temp_fpr = FprFunctionVaried(num_layers + 1, temp_rank_and_lfprs.data(), nullptr, &fpr_params);
                if (temp_fpr < current_minimum_fpr) {
                    rank_and_layer_fprs = temp_rank_and_lfprs;
                    current_minimum_fpr = temp_fpr;
                }
            } else {
                double temp_size = SizeFunctionVaried(num_layers + 1, temp_rank_and_lfprs.data(), nullptr,
                                                      &size_params);
                if (temp_size < current_minimum_size) {
                    rank_and_layer_fprs = temp_rank_and_lfprs;
                    current_minimum_size = temp_size;
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
            StackedAMQ<CQFilter, IntElement> filter(layer_fprs, positives, known_negatives);
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
                double uniform_double = dist(gen);
                uint64 element_rank = inverseCdfFast(uniform_double, zipf_parameter, negative_universe_size);
                negatives_to_test.push_back(negatives[element_rank]);
                if (element_rank < num_known_negative_elements) {
                    known_negatives_tested++;
                } else {
                    unknown_negatives_tested++;
                }
            }
            auto negative_lookup_start = std::chrono::system_clock::now();
            for (uint64 i = 0; i < negative_sample_size; i++) {
                false_positives += filter.LookupElement(negatives_to_test[i]);
            }
            auto negative_lookup_end = std::chrono::system_clock::now();

            const auto negative_lookup_rep_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    negative_lookup_end - negative_lookup_start);
            negative_lookup_time += negative_lookup_rep_time.count() / 1000000.0;

            checks_per_neg +=
                    (double) filter.NumFilterChecks() / negative_sample_size;
            filter.ResetNumFilterChecks();

            known_fpr += (double) (known_false_positives) / (double) (known_negatives_tested);
            unknown_fpr +=
                    (double) (unknown_false_positives) / (double) (unknown_negatives_tested);
            total_fpr += (double) false_positives / (double) negative_sample_size;
            psi += (double) known_negatives_tested / negative_sample_size;
            printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f\n",
                   (double) false_positives / (double) negative_sample_size,
                   num_known_negative_elements, (double) known_negatives_tested / negative_sample_size);
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
                    << checks_per_neg << "\n";
        printf(", fpr=%f, pos_checks=%f, neg_checks=%f, construction_time=%f, positive_lookup_time=%f, negative_lookup_time=%f",
               total_fpr,
               checks_per_pos, checks_per_neg, construction_time, positive_lookup_time, negative_lookup_time);
        printf(" Used Bits= %f\n", used_bits);
    }

    void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
                   double &bits_max, double &bits_step, double &fpr_begin,
                   double &fpr_min, double &fpr_ratio, uint64 &neg_universe_size,
                   uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_known_negatives,
                   uint64 &max_known_negatives_max, double &max_known_negatives_ratio, bool &equal_fprs,
                   uint32 &num_layers,
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
              neg_universe_size_ratio, max_known_negatives_begin, max_known_negatives_max,
              max_known_negatives_ratio,
              equal_fprs, num_layers,
              num_reps, file_path, arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Number Of Positive Elements,Negative Sample Size,Zipf Parameter,Negative Universe Size,Number of Known Negatives Available,Number of Known Negatives Chosen,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
               "FPR,Known FPR,Unknown FPR,Construction Time,Positive Lookup Time,Negative Lookup Time,Filter Checks For Positive,Filter Checks For "
               "Negative\n";
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
                    }

                }
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop