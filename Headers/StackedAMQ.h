#pragma once

#include <vector>
#include <random>
#include "BloomFilter.h"
#include "CQFilter.h"
#include "CuckooFilter.h"
#include "Common.h"
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"
#include "nlopt.h"
#include "OptimizationRoutines.h"


template<template<typename> class BaseAMQ, typename element_type>
class StackedAMQ {
public:
    unsigned int num_layers_;
    double fpr_;
    double beta_;
    double psi_;
    double penalty_coef_;
    size_t num_positive_;
    size_t num_negative_;
    size_t total_size_;
    bool equal_layer_fprs_ = false;
    std::vector<BaseAMQ<element_type>> layer_array_;
    std::vector<double> layer_fprs_;

    StackedAMQ(uint32 num_layers,
               const std::vector<element_type> &positives,
               const std::vector<element_type> &negatives,
               size_t total_size, double psi,
               double penalty_coef, bool equal_layer_fprs);

    StackedAMQ(const size_t total_size,
               const std::vector<element_type> &positives,
               const std::vector<element_type> &negatives,
               const std::vector<double> &pmf);

    StackedAMQ(const std::vector<double> &layer_fprs,
               const std::vector<element_type> &positives,
               const std::vector<element_type> &negatives,
               const double insert_capacity = 0);

    StackedAMQ(const std::vector<double> &layer_fprs,
               const std::vector<uint32> &integral_parameters,
               const std::vector<element_type> &positives,
               const std::vector<element_type> &negatives);

    ~StackedAMQ();

    void InitStackedAMQ(const std::vector<double> &layer_fprs,
                        std::vector<uint32> integral_parameters,
                        const std::vector<element_type> &positives,
                        const std::vector<element_type> &negatives,
                        const double insert_capacity = 0);

    bool LookupElement(const element_type element);

    void InsertPositiveElement(const element_type element);

    void DeleteElement(element_type element);

    size_t GetSize();

    std::vector<double> CalculateLayerFPRs();

    size_t NumFilterChecks();

    void ResetNumFilterChecks();

    void PrintLayerDiagnostics();

    static double FprFunctionVaried(unsigned num_layers, const double *layer_fprs,
                                    double *grad, void *filter_ptr) {
        StackedAMQ<BaseAMQ, element_type> *filter =
                (StackedAMQ<BaseAMQ, element_type> *) filter_ptr;
        double psi = filter->psi_;
        double known_fpr = layer_fprs[0];
        double penalty_coef = filter->penalty_coef_;
        for (int i = 1; i <= (num_layers - 1) / 2; i++) {
            known_fpr = known_fpr * layer_fprs[2 * i];
        }
        double unknown_fpr_side = 0;
        for (int i = 1; i <= (num_layers - 1) / 2; i++) {
            double temp_fpr = layer_fprs[0];
            for (int j = 1; j <= 2 * (i - 1); j++) {
                temp_fpr = temp_fpr * layer_fprs[j];
            }
            unknown_fpr_side += temp_fpr * (1 - layer_fprs[2 * i - 1]);
        }
        double unknown_fpr_end = layer_fprs[0];
        for (unsigned int i = 1; i <= (num_layers - 1) / 2; i++) {
            unknown_fpr_end = unknown_fpr_end * layer_fprs[i];
        }
        double total_fpr =
                psi * known_fpr + (1 - psi) * (unknown_fpr_side + unknown_fpr_end);
        // Penalty Function For Number of Hashes
        int num_hashes = 0;
        for (int i = 0; i < num_layers; i++) num_hashes += std::max((int) (round(-log(layer_fprs[i]) / log(2))), 1);
        return (psi * known_fpr +
                (1 - psi) * (unknown_fpr_side + unknown_fpr_end)) *
               (1 + num_hashes * penalty_coef);
    };

    static double FprFunctionEqual(unsigned num_layers, const double *layer_fprs,
                                   double *grad, void *filter_ptr) {
        StackedAMQ<BaseAMQ, element_type> *filter =
                (StackedAMQ<BaseAMQ, element_type> *) filter_ptr;
        double psi = filter->psi_;
        const double layer_fpr = layer_fprs[0];
        double known_fpr = layer_fpr;
        double penalty_coef = filter->penalty_coef_;
        for (int i = 1; i <= (num_layers - 1) / 2; i++) {
            known_fpr = known_fpr * layer_fpr;
        }
        double unknown_fpr_side = 0;
        for (int i = 1; i <= (num_layers - 1) / 2; i++) {
            double temp_fpr = layer_fpr;
            for (int j = 1; j <= 2 * (i - 1); j++) {
                temp_fpr = temp_fpr * layer_fpr;
            }
            unknown_fpr_side += temp_fpr * (1 - layer_fpr);
        }
        double unknown_fpr_end = layer_fpr;
        for (int i = 1; i < num_layers; i++) {
            unknown_fpr_end = unknown_fpr_end * layer_fpr;
        }
        int num_hashes = num_layers * std::max((int) (round(-log(layer_fpr) / log(2))), 1);
        return (psi * known_fpr +
                (1 - psi) * (unknown_fpr_side + unknown_fpr_end)) *
               (1 + num_hashes * penalty_coef);
    };

    static double SizeFunctionVaried(unsigned num_layers, const double *layer_fprs,
                                     double *grad, void *filter_ptr) {
        StackedAMQ<BaseAMQ, element_type> *filter =
                (StackedAMQ<BaseAMQ, element_type> *) filter_ptr;
        size_t total_size = filter->total_size_ * .995;
        size_t num_positive = filter->num_positive_;
        size_t num_negative = filter->num_negative_;
        double size = 0;
        double positive_fpr = 1;
        double negative_fpr = 1;
        for (unsigned int i = 0; i < num_layers; i++) {
            int num_hashes = std::max<int>(ceil(-log2(layer_fprs[i])), 1);
            double temp_size;
            if (((i + 2) % 2) == 0) {
                temp_size = BaseAMQ<element_type>::SizeFunction(
                        layer_fprs[i], num_positive * positive_fpr);
                negative_fpr *= layer_fprs[i];
            } else {
                temp_size = BaseAMQ<element_type>::SizeFunction(
                        layer_fprs[i], num_negative * negative_fpr);
                positive_fpr *= layer_fprs[i];
            }
            // FPR formulas often do not work well on small filters, so we put
            // a floor on the size of each layer.
            if (temp_size < 2000) temp_size = 2000;
            size += temp_size;
        }
        return size - total_size;
    };

    static double SizeFunctionEqual(unsigned num_layers, const double *layer_fprs,
                                    double *grad, void *filter_ptr) {
        StackedAMQ<BaseAMQ, element_type> *filter =
                (StackedAMQ<BaseAMQ, element_type> *) filter_ptr;
        size_t total_size = filter->total_size_ * .999;
        size_t num_positive = filter->num_positive_;
        size_t num_negative = filter->num_negative_;
        double size = 0;
        const double layer_fpr = layer_fprs[0];
        double positive_fpr = 1;
        double negative_fpr = 1;
        for (unsigned int i = 0; i < num_layers; i++) {
            int num_hashes = std::max<int>(ceil(-log2(layer_fprs[i])), 1);
            if ((i % 2) == 0) {
                size += BaseAMQ<element_type>::SizeFunction(
                        layer_fpr, positive_fpr * num_positive);
                negative_fpr *= layer_fpr;
            } else {
                size += BaseAMQ<element_type>::SizeFunction(
                        layer_fpr, negative_fpr * num_negative);
                positive_fpr *= layer_fpr;
            }
        }
        return size - total_size;
    };
};

template
class StackedAMQ<BloomFilter, IntElement>;

template
class StackedAMQ<BloomFilter, StringElement>;

template
class StackedAMQ<BloomFilter, BigIntElement>;

template
class StackedAMQ<CQFilter, IntElement>;

template
class StackedAMQ<CuckooFilterLayer, IntElement>;
