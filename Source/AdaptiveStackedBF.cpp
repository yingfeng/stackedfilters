#include "../Headers/AdaptiveStackedBF.h"

// Calculate the proper layer FPRs and sample size then hand off to the constructor that takes in
// layer fprs.
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
    layer_array_ = std::vector<BloomFilter<element_type>>();
    layer_array_.reserve(num_layers_);

    // Build the first layer.
    auto first_layer_size = BloomFilter<element_type>::SizeFunction(layer_fprs_[0], num_positive_);
    layer_array_.emplace_back(first_layer_size, integral_parameters_[0], random_gen_(eng_));
    total_size_ += first_layer_size;
    for (auto element : positives) layer_array_[0].InsertElement(element);

    // Reserve the second layer.
    auto second_layer_size = BloomFilter<element_type>::SizeFunction(layer_fprs_[1], false_positives_capacity_);
    layer_array_.emplace_back(second_layer_size, integral_parameters_[1], random_gen_(eng_));
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
    auto third_layer_size = BloomFilter<element_type>::SizeFunction(layer_fprs_[2], false_negatives.size());
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
        uint64_t second_layer_size = BloomFilter<IntElement>::SizeFunction(layer_fprs_[1], false_positives_capacity_);
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
    double *collection_psi;
    size_t *collection_size;
    ASFContinuousOptimizationObject opt_result = optimizeASFContinuous(total_size_ / static_cast<double>(num_positive_),
                                                                       num_positive_, total_queries_,
                                                                       cdf, true);
    integral_parameters_.clear();
    layer_fprs_.clear();
    for(auto bits : opt_result.bitsPerElementLayers){
        int num_hashes = round(bits*log(2.));
        integral_parameters_.push_back(num_hashes);
        double fpr = pow(2., -num_hashes);
        layer_fprs_.push_back(fpr);
    }
    // Set the threshold FPR to be halfway between the cold and warm FPR.
    target_fpr_ = layer_fprs_[0] *
                  ((1 - opt_result.psi) * ((1 - layer_fprs_[1]) + layer_fprs_[1] * layer_fprs_[2]) + opt_result.psi * layer_fprs_[2]);
    const double single_layer_fpr = exp(-static_cast<double>(total_size_) / num_positive_ * log(2) * log(2));
    threshold_fpr_ = -1;
    // Adjust for the fact that only elements which flip at least one bit will count towards the capacity
    size_t negatives_size = opt_result.NoverP * num_positive_;
    std::cout << opt_result.NoverP << std::endl;
    double capacity_adjustment = (1 - layer_fprs_[1]);
    false_positives_capacity_ = negatives_size * layer_fprs_[0] * capacity_adjustment;
    free(collection_psi);
    free(collection_size);
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
