#pragma once

/*
 * ============================================================================
 *        Stacked Filters: Learning to Filter by Structure
 *        Authors:  Kyle Deeds <kdeeds@cs.washington.edu>
 *                  Brian Hentschel <bhentschel@g.harvard.edu>
 *                  Stratos Idreos <stratos@seas.harvard.edu>
 *
 * ============================================================================
 */


#include <vector>
#include <random>
#include "BloomFilterLayer.h"
#include "../QuotientFilter/CQFilterLayer.h"
#include "../CuckooFilter/CuckooFilterLayer.h"
#include "Common.h"
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"
#include "../Optimization/OptimizationRoutines.h"


/*
 * Stacked Filters perform the same function as traditional filters such as bloom filters, but utilize additional
 * information about the workload to achieve higher performance. They expose just a few functions: bulk construction,
 * lookup, insertion, and deletion (if the base filter type allows deletion). The bulk constructor is where the additional
 * workload knowledge is included in the form of an array of "true negative" elements and their corresponding query
 * frequency CDF.
 */
template<template<typename> class LayerType, typename element_type>
class StackedFilter {
public:
    unsigned int num_layers_;
    double psi_;
    double penalty_coef_;
    size_t num_positive_;
    size_t num_negative_;
    size_t total_size_;
    std::vector<LayerType<element_type>> layer_array_;
    std::vector<double> layer_fprs_;

    StackedFilter(const size_t total_size,
                  const std::vector<element_type> &positives,
                  const std::vector<element_type> &negatives,
                  const std::vector<double> &cdf,
                  const double insert_capacity = 0);

    StackedFilter(const std::vector<double> &layer_fprs,
                  const std::vector<element_type> &positives,
                  const std::vector<element_type> &negatives,
                  const double insert_capacity = 0);

    StackedFilter(const std::vector<double> &layer_fprs,
                  const std::vector<uint32> &integral_parameters,
                  const std::vector<element_type> &positives,
                  const std::vector<element_type> &negatives);

    ~StackedFilter();


    bool LookupElement(const element_type element);

    void InsertPositiveElement(const element_type element);

    void DeleteElement(element_type element);

    size_t GetSize();

    size_t NumFilterChecks();

    void ResetNumFilterChecks();

    void PrintLayerDiagnostics();

private:
    void InitStackedAMQ(const std::vector<double> &layer_fprs,
                        std::vector<uint32> integral_parameters,
                        const std::vector<element_type> &positives,
                        const std::vector<element_type> &negatives,
                        const double insert_capacity = 0);
};


// Calculate the proper layer FPRs then hand off to the constructor that takes in
// layer fprs.
template<template<typename> class BaseAMQ, typename element_type>
StackedFilter<BaseAMQ, element_type>::StackedFilter(const size_t total_size,
                                                    const std::vector<element_type> &positives,
                                                    const std::vector<element_type> &negatives,
                                                    const std::vector<double> &cdf,
                                                    const double insert_capacity) {
    static constexpr double kEpsilonError = .000001;
    static constexpr double kLoadFactor = .95;
    std::vector<double> layer_fprs;
    std::vector<unsigned int> integral_parameters;
    uint64_t num_chosen_negatives = 0;
    if (std::is_same<BaseAMQ<element_type>, BloomFilterLayer<element_type>>::value) {
        std::tie(num_chosen_negatives, layer_fprs) = optimizeStackedFilterBloom(
                static_cast<double>(total_size) / positives.size(), positives.size() * (1 + insert_capacity),
                kEpsilonError, cdf);
        for (auto layer_fpr : layer_fprs) {
            int num_hashes = round(-log2(layer_fpr));
            integral_parameters.push_back(num_hashes);
        }
    } else {
        std::vector<int> fingerprint_bits;
        std::tie(num_chosen_negatives, fingerprint_bits) = optimizeDiscreteStackedFilter(
                static_cast<double>(total_size) / positives.size(), positives.size() * (1 + insert_capacity),
                kEpsilonError, cdf, kLoadFactor);
        for (auto bits : fingerprint_bits) {
            double fpr = powf(2., -bits);
            integral_parameters.push_back(bits);
            layer_fprs.push_back(fpr);
        }
    }
    num_chosen_negatives = std::min(negatives.size(), num_chosen_negatives);
    std::vector<element_type> chosen_negatives(negatives.begin(), negatives.begin() + num_chosen_negatives);
    InitStackedAMQ(layer_fprs, integral_parameters, positives, chosen_negatives, insert_capacity);
}

// Allows the caller to calculate the proper layer fprs.
template<template<typename> class BaseAMQ, typename element_type>
StackedFilter<BaseAMQ, element_type>::StackedFilter(const std::vector<double> &layer_fprs,
                                                    const std::vector<element_type> &positives,
                                                    const std::vector<element_type> &negatives,
                                                    const double insert_capacity) {
    InitStackedAMQ(layer_fprs, {}, positives, negatives, insert_capacity);
}


template<template<typename> class BaseAMQ, typename element_type>
StackedFilter<BaseAMQ, element_type>::StackedFilter(const std::vector<double> &layer_fprs,
                                                    const std::vector<uint32> &integral_parameters,
                                                    const std::vector<element_type> &positives,
                                                    const std::vector<element_type> &negatives) {
    InitStackedAMQ(layer_fprs, integral_parameters, positives, negatives);
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedFilter<BaseAMQ, element_type>::InitStackedAMQ(const std::vector<double> &layer_fprs,
                                                          std::vector<uint32> integral_parameters,
                                                          const std::vector<element_type> &positives,
                                                          const std::vector<element_type> &negatives,
                                                          const double insert_capacity) {

    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long long> random_gen;

    num_layers_ = layer_fprs.size();
    if (integral_parameters.empty()) {
        for (int i = 0; i < num_layers_; i++)
            integral_parameters.push_back(std::max<int>(ceil(-log2(layer_fprs[i])), 1));
    }
    num_positive_ = positives.size();
    num_negative_ = negatives.size();
    total_size_ = 0;
    layer_fprs_ = layer_fprs;
    layer_array_ = std::vector<BaseAMQ<element_type>>();
    layer_array_.reserve(num_layers_);

    // Build the first layer.
    auto size = BaseAMQ<element_type>::SizeFunction(layer_fprs_[0], num_positive_ * (1 + insert_capacity));
    if (size < 2000) size = 2000;
    layer_array_.emplace_back(size, integral_parameters[0], random_gen(eng));
    total_size_ += size;
    for (auto element : positives) layer_array_[0].InsertElement(element);

    if (num_layers_ == 1) {
        return;
    }

    // Add the elements to the filters in a descending order by layer.
    // Build the first and second layer first in order to avoid copying the full negative and positive vectors.
    std::vector<element_type> negative_fp;
    negative_fp.reserve(layer_fprs_[0] * negatives.size() * 1.1);
    std::vector<element_type> positive_fp;
    positive_fp.reserve(layer_fprs_[1] * positives.size() * 1.1);
    size_t num_negative_fp = 0;
    size_t num_positive_fp = 0;

    for (auto element : negatives) {
        if (layer_array_[0].LookupElement(element)) {
            negative_fp.push_back(element);
            num_negative_fp++;
        }
    }
    size = BaseAMQ<element_type>::SizeFunction(layer_fprs_[1], num_negative_fp);
    if (size < 2000) size = 2000;
    layer_array_.emplace_back(size, integral_parameters[1], random_gen(eng));
    total_size_ += size;
    for (auto element : negative_fp) layer_array_[1].InsertElement(element);

    for (auto element : positives) {
        if (layer_array_[1].LookupElement(element)) {
            positive_fp.push_back(element);
            num_positive_fp++;
        }
    }

    for (int i = 2; i < num_layers_; i++) {
        // Calculate the layer's size and allocate it.
        if ((i % 2) == 0) {
            size = BaseAMQ<element_type>::SizeFunction(layer_fprs_[i], num_positive_fp * (1 + insert_capacity));
        } else {
            size = BaseAMQ<element_type>::SizeFunction(layer_fprs_[i], num_negative_fp);
        }
        // Bloom Filter FPR formulas do not work well on small filters, so we put a
        // floor on the size of each layer.
        if (size < 2000) size = 2000;
        layer_array_.emplace_back(size, integral_parameters[i], random_gen(eng));
        total_size_ += size;


        if ((i % 2) == 0) {
            for (auto element : positive_fp) layer_array_[i].InsertElement(element);
            int temp_neg_fp = 0;
            for (auto element : negative_fp) {
                if (layer_array_[i].LookupElement(element)) {
                    negative_fp[temp_neg_fp] = element;
                    temp_neg_fp++;
                }
            }
            num_negative_fp = temp_neg_fp;
            negative_fp.resize(num_negative_fp);
        } else {
            for (auto element : negative_fp) layer_array_[i].InsertElement(element);
            int temp_pos_fp = 0;
            for (auto element : positive_fp) {
                if (layer_array_[i].LookupElement(element)) {
                    positive_fp[temp_pos_fp] = element;
                    temp_pos_fp++;
                }
            }
            num_positive_fp = temp_pos_fp;
            positive_fp.resize(num_positive_fp);
        }
    }
}

template<template<typename> class BaseAMQ, typename element_type>
StackedFilter<BaseAMQ, element_type>::~StackedFilter() {}

_GLIBCXX17_INLINE
template<template<typename> class BaseAMQ, typename element_type>
bool StackedFilter<BaseAMQ, element_type>::LookupElement(const element_type element) {
    for (int i = 0; i < num_layers_; i++) {
        if (layer_array_[i].LookupElement(element) == false) return i % 2 != 0;
    }
    return true;
}

_GLIBCXX17_INLINE
template<template<typename> class BaseAMQ, typename element_type>
void StackedFilter<BaseAMQ, element_type>::InsertPositiveElement(
        element_type element) {
    layer_array_[0].InsertElement(element);
    for (int i = 0; i < (num_layers_ - 1) / 2; i++) {
        if (layer_array_[2 * i + 1].LookupElement(element) == true) {
            layer_array_[2 * i + 2].InsertElement(element);
        } else {
            return;
        }
    }
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedFilter<BaseAMQ, element_type>::DeleteElement(element_type element) {}


// Optimization

template<template<typename> class BaseAMQ, typename element_type>
size_t StackedFilter<BaseAMQ, element_type>::GetSize() {
    size_t size = 0;
    for (int i = 0; i < num_layers_; i++) {
        size += layer_array_[i].GetSize();
    }
    return size;
}

template<template<typename> class BaseAMQ, typename element_type>
size_t StackedFilter<BaseAMQ, element_type>::NumFilterChecks() {
    size_t num_filter_checks = 0;
    for (int i = 0; i < num_layers_; i++) {
        num_filter_checks += layer_array_[i].num_checks_;
    }
    return num_filter_checks;
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedFilter<BaseAMQ, element_type>::ResetNumFilterChecks() {
    for (int i = 0; i < num_layers_; i++) layer_array_[i].num_checks_ = 0;
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedFilter<BaseAMQ, element_type>::PrintLayerDiagnostics() {
    for (int i = 0; i < num_layers_; i++) {
        printf(
                "Layer %d FPR: %.20f Size:%ld Num element_types:%ld Load "
                "Factor:%f \n",
                i, layer_fprs_[i], layer_array_[i].GetSize(), layer_array_[i].GetNumElements(),
                layer_array_[i].GetLoadFactor());
    }
}


template
class StackedFilter<BloomFilterLayer, IntElement>;

template
class StackedFilter<BloomFilterLayer, StringElement>;

template
class StackedFilter<CQFilterLayer, IntElement>;

template
class StackedFilter<CuckooFilterLayer, IntElement>;
