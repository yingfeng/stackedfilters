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
#include <iterator>
#include <experimental/algorithm>

#include "BloomFilterLayer.h"
#include "../QuotientFilter/CQFilterLayer.h"
#include "../CuckooFilter/CuckooFilterLayer.h"
#include "Common.h"
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"
#include "../Optimization/ASFContinuousOptimizationRoutines.h"


/*
 * Adaptive Stacked Filters slowly obtain workload information over time to improve the efficacy of the filter. In order
 * to do this, the user must declare false positives to the filter as they come up and respond to the status of the
 * filter. This status lets the user know when to provide the filter with access to the positive set so that it can
 * build or rebuild certain layers through the functions BuildThirdLayer and RebuildFilter.
 */
enum FilterStatus {
    OPERATIONAL = 0,
    NEEDS_THIRD_LAYER = 1,
    NEEDS_REBUILD = 2
};

template<typename element_type>
class AdaptiveStackedBF {
public:
    static constexpr double penalty_coef_ = .0005;
    static constexpr unsigned int num_layers_ = 3;

    bool fully_built_ = false;
    size_t num_positive_;
    size_t total_size_;
    size_t sample_estimate_size_;
    size_t total_queries_;
    size_t false_positives_capacity_;
    size_t num_false_positives_inserted_ = 0;
    std::vector<BloomFilterLayer<element_type>> layer_array_;
    std::vector<double> layer_fprs_;
    std::vector<uint64_t> integral_parameters_;


    // Performance Monitoring Statistics
    double target_fpr_;
    double threshold_fpr_;
    size_t false_positives_seen_ = 0;
    size_t negatives_seen_ = 0;
    size_t queries_left_;
    FilterStatus current_status_ = OPERATIONAL;

    std::random_device rd_;
    std::mt19937_64 eng_;
    std::uniform_int_distribution<unsigned long long> random_gen_;


    AdaptiveStackedBF(
            const std::vector<element_type> &positives,
            size_t total_size, size_t total_queries, const std::vector<double> &cdf);

    void InitAdaptiveStackedBF(const std::vector<element_type> &positives);

    ~AdaptiveStackedBF();

    bool LookupElement(element_type element);

    void InsertPositive(element_type element);

    FilterStatus DeclareFalsePositiveAndCheckStatus(element_type element);

    void BuildThirdLayer(const std::vector<element_type> &positives);

    void RebuildFilter(const std::vector<element_type> &positives = std::vector<element_type>(),
                       const std::vector<double> &cdf = std::vector<double>());

    void DeleteElement(element_type element);

    size_t GetSize();

    void CalculateLayerFPRsAndFPCapacity(const std::vector<double> &cdf);

    size_t NumFilterChecks();

    void ResetNumFilterChecks();

    void PrintLayerDiagnostics();
};

template<typename element_type>
AdaptiveStackedBF<element_type>::AdaptiveStackedBF(
        const std::vector<element_type> &positives,
        const size_t total_size, const size_t total_queries, const std::vector<double> &cdf) {
    num_positive_ = positives.size();
    total_size_ = total_size;
    total_queries_ = total_queries;
    queries_left_ = total_queries;
    eng_ = std::mt19937_64(rd_());

    CalculateLayerFPRsAndFPCapacity(cdf);
    InitAdaptiveStackedBF(positives);
}

template<typename element_type>
void AdaptiveStackedBF<element_type>::InitAdaptiveStackedBF(const std::vector<element_type> &positives) {
    integral_parameters_ = std::vector<uint64_t>();
    for (int i = 0; i < num_layers_; i++)
        integral_parameters_.push_back(std::max<int>(ceil(-log2(layer_fprs_[i])), 1));
    num_positive_ = positives.size();
    total_size_ = 0;
    layer_array_ = std::vector<BloomFilterLayer<element_type>>();
    layer_array_.reserve(num_layers_);

    // Build the first layer.
    auto first_layer_size = BloomFilterLayer<element_type>::SizeFunction(layer_fprs_[0], num_positive_);
    layer_array_.emplace_back(first_layer_size, integral_parameters_[0], random_gen_(eng_));
    total_size_ += first_layer_size;
    for (auto element : positives) layer_array_[0].InsertElement(element);

    // Reserve the second layer.
    auto second_layer_size = BloomFilterLayer<element_type>::SizeFunction(layer_fprs_[1], false_positives_capacity_);
    layer_array_.emplace_back(second_layer_size, integral_parameters_[1], random_gen_(eng_));

    std::cout << "Second Layer Size: " << second_layer_size << " False Positives Capacity: " << false_positives_capacity_ << std::endl;
}

template<typename element_type>
AdaptiveStackedBF<element_type>::~AdaptiveStackedBF() {}

_GLIBCXX17_INLINE
template<typename element_type>
bool AdaptiveStackedBF<element_type>::LookupElement(const element_type element) {
    queries_left_--;
    bool layer_one_positive = layer_array_[0].LookupElement(element);
    if (!layer_one_positive || !fully_built_) {
        negatives_seen_ += 1 - layer_one_positive;
        return layer_one_positive;
    }
    bool layer_two_positive = layer_array_[1].LookupElement(element);
    if (!layer_two_positive) return true;
    bool layer_three_positive = layer_array_[2].LookupElement(element);
    negatives_seen_ += 1 - layer_three_positive;
    return layer_three_positive;
}

_GLIBCXX17_INLINE
template<typename element_type>
void AdaptiveStackedBF<element_type>::InsertPositive(const element_type element) {
    layer_array_[0].InsertElement(element);
    if (!fully_built_) return;
    if (layer_array_[1].LookupElement(element)) layer_array_[2].InsertElement(element);
}

_GLIBCXX17_INLINE
template<typename element_type>
FilterStatus AdaptiveStackedBF<element_type>::DeclareFalsePositiveAndCheckStatus(const element_type element) {
    bool has_space = num_false_positives_inserted_ < false_positives_capacity_;
    if (has_space) {
        current_status_ = OPERATIONAL;
        if (layer_array_[1].LookupElement(element)) {
            return current_status_;
        }
        layer_array_[1].InsertElement(element);
        num_false_positives_inserted_++;
        return current_status_;
    }
    if (!fully_built_) {
        current_status_ = NEEDS_THIRD_LAYER;
        return current_status_;
    }

    // Monitor the false positive rate and signal a rebuild if it's too high.
    false_positives_seen_++;
    if (false_positives_seen_ + negatives_seen_ >= 100000) {
        double false_positive_rate =
                static_cast<double>(false_positives_seen_) / (negatives_seen_ + false_positives_seen_);
        false_positives_seen_ = 0;
        negatives_seen_ = 0;
        if (threshold_fpr_ == -1) {
            threshold_fpr_ = false_positive_rate * 1.5;
            current_status_ = OPERATIONAL;
            return current_status_;
        }
        if (false_positive_rate > threshold_fpr_) {
            current_status_ = NEEDS_REBUILD;
        }
    }
    return current_status_;
}

_GLIBCXX17_INLINE
template<typename element_type>
void AdaptiveStackedBF<element_type>::BuildThirdLayer(const std::vector<element_type> &positives) {
    std::vector<element_type> false_negatives;
    for (const auto &element : positives) {
        if (layer_array_[1].LookupElement(element)) {
            false_negatives.push_back(element);
        }
    }
    auto third_layer_size = BloomFilterLayer<element_type>::SizeFunction(layer_fprs_[2], false_negatives.size());
    layer_array_.emplace_back(third_layer_size, integral_parameters_[2], random_gen_(eng_));
    for (const auto &element : false_negatives) {
        layer_array_[2].InsertElement(element);
    }
    fully_built_ = true;
    threshold_fpr_ = -1;
    false_positives_seen_ = 0;
    negatives_seen_ = 0;
    current_status_ = OPERATIONAL;
}

template<typename element_type>
void AdaptiveStackedBF<element_type>::RebuildFilter(const std::vector<element_type> &positives,
                                                    const std::vector<double> &cdf) {
    // Clear the second and third filters
    fully_built_ = false;
    threshold_fpr_ = -1;
    num_false_positives_inserted_ = 0;
    layer_array_[1].filter_.resize(0);
    layer_array_.resize(2);
    // If a full rebuild is not requested, the second layer is simply reserved again.
    if (positives.empty() || cdf.empty()) {
        uint64_t second_layer_size = BloomFilterLayer<IntElement>::SizeFunction(layer_fprs_[1], false_positives_capacity_);
        layer_array_[1].filter_.resize(second_layer_size, false);
        current_status_ = OPERATIONAL;
        return;
    }
    // For a full rebuild, the optimization is re-run, and the first layer is rebuilt.
    CalculateLayerFPRsAndFPCapacity(cdf);
    InitAdaptiveStackedBF(positives);
    current_status_ = OPERATIONAL;
}

template<typename element_type>
void AdaptiveStackedBF<element_type>::DeleteElement(element_type element) {}

// Use NLOPT to calculate layer fprs and the number of unique false positives to be included.
template<typename element_type>
void AdaptiveStackedBF<element_type>::CalculateLayerFPRsAndFPCapacity(const std::vector<double> &cdf) {
    std::cout <<"Bits Available: " << total_size_ / static_cast<double>(num_positive_) << " Num Positives: " << num_positive_ << " Total Queries: " << total_queries_ <<std::endl;
    ASFContinuousOptimizationObject opt_result = optimizeASFContinuous(total_size_ / static_cast<double>(num_positive_),
                                                                       num_positive_, total_queries_,
                                                                       cdf, true);
    integral_parameters_.clear();
    layer_fprs_.clear();
    for(auto bits : opt_result.bitsPerElementLayers){
        int num_hashes = round(bits*log(2.));
        integral_parameters_.push_back(num_hashes);
        double fpr = exp(-bits*log(2.)*log(2.));
        layer_fprs_.push_back(fpr);
    }
    // Set the threshold FPR to be halfway between the cold and warm FPR.
    target_fpr_ = layer_fprs_[0] *
                  ((1 - opt_result.psi) * ((1 - layer_fprs_[1]) + layer_fprs_[1] * layer_fprs_[2]) + opt_result.psi * layer_fprs_[2]);
    threshold_fpr_ = -1;
    // Adjust for the fact that only elements which flip at least one bit will count towards the capacity
    size_t negatives_size = opt_result.NoverP * num_positive_ / layer_fprs_[0];
    double capacity_adjustment = (1 - layer_fprs_[1]);
    false_positives_capacity_ = negatives_size * layer_fprs_[0] * capacity_adjustment;
}


/* Diagnostic Methods */

template<typename element_type>
size_t AdaptiveStackedBF<element_type>::GetSize() {
    size_t size = 0;
    if (fully_built_) {
        for (int i = 0; i < num_layers_; i++) {
            size += layer_array_[i].GetSize();
        }
    } else {
        size += layer_array_[0].GetSize();
        size += layer_array_[1].GetSize();
    }
    return size;
}

template<typename element_type>
size_t AdaptiveStackedBF<element_type>::NumFilterChecks() {
    size_t num_filter_checks = 0;
    for (int i = 0; i < num_layers_; i++) {
        num_filter_checks += layer_array_[i].num_checks_;
    }
    return num_filter_checks;
}

template<typename element_type>
void AdaptiveStackedBF<element_type>::ResetNumFilterChecks() {
    for (int i = 0; i < num_layers_; i++) layer_array_[i].num_checks_ = 0;
}

template<typename element_type>
void AdaptiveStackedBF<element_type>::PrintLayerDiagnostics() {
    if (!fully_built_) {
        for (int i = 0; i < num_layers_ - 1; i++) {
            printf(
                    "Layer %d FPR: %.20f Size:%ld Num element_types:%ld Load "
                    "Factor:%f \n",
                    i, layer_fprs_[i], layer_array_[i].GetSize(), layer_array_[i].GetNumElements(),
                    layer_array_[i].GetLoadFactor());
        }
        printf("Layer %d FPR: %.20f  Size:0 Num element_types:0 Load Factor:0\n", 2, layer_fprs_[2]);
        return;
    }
    for (int i = 0; i < num_layers_; i++) {
        printf(
                "Layer %d FPR: %.20fSize:%ld Num element_types:%ld Load "
                "Factor:%f \n",
                i, layer_fprs_[i], layer_array_[i].GetSize(), layer_array_[i].GetNumElements(),
                layer_array_[i].GetLoadFactor());
    }
}





template
class AdaptiveStackedBF<IntElement>;

template
class AdaptiveStackedBF<StringElement>;

template
class AdaptiveStackedBF<BigIntElement>;