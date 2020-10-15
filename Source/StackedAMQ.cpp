#include "StackedAMQ.h"

// Use NLOPT to calculate the proper layer FPRs then hand off to the constructor that takes in
// layer fprs.
template<template<typename> class BaseAMQ, typename element_type>
StackedAMQ<BaseAMQ, element_type>::StackedAMQ(
        const uint32 num_layers, const std::vector<element_type> &positives,
        const std::vector<element_type> &negatives, const size_t total_size,
        const double psi, const double penalty_coef, const bool equal_layer_fprs) {
    penalty_coef_ = penalty_coef;
    equal_layer_fprs_ = equal_layer_fprs;
    num_layers_ = num_layers;
    num_positive_ = positives.size();
    num_negative_ = negatives.size();
    total_size_ = total_size;
    psi_ = psi;
    beta_ = (double) num_positive_ / (double) (num_positive_ + num_negative_);
    layer_fprs_ = CalculateLayerFPRs();
    InitStackedAMQ(layer_fprs_, {}, positives, negatives);
}


template<template<typename> class BaseAMQ, typename element_type>
StackedAMQ<BaseAMQ, element_type>::StackedAMQ(const size_t total_size,
                                              const std::vector<element_type> &positives,
                                              const std::vector<element_type> &negatives,
                                              const std::vector<double> &pmf) {
    static constexpr double kEpsilonError = .0005;
    std::vector<double> layer_fprs;
    std::vector<int> integral_parameters;
    if(std::is_same<BaseAMQ<element_type>, BloomFilter<element_type>>::value){
        std::vector<double> bits_per_element = optimizeStackedFilterBloom(total_size, positives.size(), kEpsilonError, pmf);
        for(auto bits : bits_per_element){
            int num_hashes = round(bits*log(2.));
            double fpr = exp(-bits*log(2.)*log(2.));
            integral_parameters.push_back(num_hashes);
            layer_fprs.push_back(fpr);
        }
    } else{
        std::vector<int> fingerprint_bits = optimizeDiscreteStackedFilter(total_size, positives.size(), kEpsilonError, pmf);
        for(auto bits : fingerprint_bits){
            double fpr =  powf(2., -bits);
            integral_parameters.push_back(bits);
            layer_fprs.push_back(fpr);
        }
    }
    InitStackedAMQ(layer_fprs, integral_parameters, positives, negatives);
}

// Allows the caller to calculate the proper layer fprs.
template<template<typename> class BaseAMQ, typename element_type>
StackedAMQ<BaseAMQ, element_type>::StackedAMQ(const std::vector<double> &layer_fprs,
                                              const std::vector<element_type> &positives,
                                              const std::vector<element_type> &negatives,
                                              const double insert_capacity) {
    InitStackedAMQ(layer_fprs, {}, positives, negatives, insert_capacity);
}


template<template<typename> class BaseAMQ, typename element_type>
StackedAMQ<BaseAMQ, element_type>::StackedAMQ(const std::vector<double> &layer_fprs,
                                              const std::vector<uint32> &integral_parameters,
                                              const std::vector<element_type> &positives,
                                              const std::vector<element_type> &negatives) {
    InitStackedAMQ(layer_fprs, integral_parameters, positives, negatives);
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedAMQ<BaseAMQ, element_type>::InitStackedAMQ(const std::vector<double> &layer_fprs,
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
    beta_ = (double) num_positive_ / (double) (num_positive_ + num_negative_);
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
    negative_fp.reserve(layer_fprs_[0] * negatives.size()*1.1);
    std::vector<element_type> positive_fp;
    positive_fp.reserve(layer_fprs_[1] * positives.size()*1.1);
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
StackedAMQ<BaseAMQ, element_type>::~StackedAMQ() {}

_GLIBCXX17_INLINE
template<template<typename> class BaseAMQ, typename element_type>
bool StackedAMQ<BaseAMQ, element_type>::LookupElement(const element_type element) {
    for (int i = 0; i < num_layers_; i++) {
        if (layer_array_[i].LookupElement(element) == false) return i % 2 != 0;
    }
    return true;
}

_GLIBCXX17_INLINE
template<template<typename> class BaseAMQ, typename element_type>
void StackedAMQ<BaseAMQ, element_type>::InsertPositiveElement(
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
void StackedAMQ<BaseAMQ, element_type>::DeleteElement(element_type element) {}


// Optimization

template<template<typename> class BaseAMQ, typename element_type>
std::vector<double> StackedAMQ<BaseAMQ, element_type>::CalculateLayerFPRs() {
    double one_level_fpr = exp(-static_cast<double>(total_size_) / num_positive_ * log(2) * log(2));
    std::vector<double> one_layer_fprs =
            std::vector<double>(num_layers_, one_level_fpr);
    for (int i = 1; i < num_layers_; i++) one_layer_fprs[i] = 1;
    if (num_layers_ == 1) {
        layer_fprs_ = one_layer_fprs;
        return layer_fprs_;
    }
    layer_fprs_ = std::vector<double>(num_layers_, .5);
    auto *zeros = (double *) calloc(num_layers_, sizeof(double));
    for (int i = 0; i < num_layers_; i++) zeros[i] = 0.00000000000000000001;
    auto *ones = (double *) calloc((num_layers_), sizeof(double));
    for (int i = 0; i < num_layers_; i++) ones[i] = 1;
    nlopt_opt local_fpr_opt = nlopt_create(NLOPT_GN_ISRES, num_layers_);
    nlopt_set_lower_bounds(local_fpr_opt, zeros);
    nlopt_set_upper_bounds(local_fpr_opt, ones);
    nlopt_set_maxtime(local_fpr_opt, .5);   // Providing more time for more parameters
    nlopt_add_inequality_constraint(
            local_fpr_opt, &StackedAMQ<BaseAMQ, element_type>::SizeFunctionVaried,
            this, total_size_ * .0005);
    nlopt_set_min_objective(local_fpr_opt,
                            &StackedAMQ<BaseAMQ, element_type>::FprFunctionVaried,
                            this);
    double variable_fpr_fpr = 1;
    nlopt_result local_ret_status =
            nlopt_optimize(local_fpr_opt, layer_fprs_.data(), &variable_fpr_fpr);
    if (local_ret_status == -4)
        printf("ERROR!!!  Roundoff Errors Reached in Local Optimization\n");
    else if (local_ret_status < 0)
        printf("ERROR!!! General Error in Local Optimization\n");
    return layer_fprs_;
}

template<template<typename> class BaseAMQ, typename element_type>
size_t StackedAMQ<BaseAMQ, element_type>::GetSize() {
    size_t size = 0;
    for (int i = 0; i < num_layers_; i++) {
        size += layer_array_[i].GetSize();
    }
    return size;
}

template<template<typename> class BaseAMQ, typename element_type>
size_t StackedAMQ<BaseAMQ, element_type>::NumFilterChecks() {
    size_t num_filter_checks = 0;
    for (int i = 0; i < num_layers_; i++) {
        num_filter_checks += layer_array_[i].num_checks_;
    }
    return num_filter_checks;
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedAMQ<BaseAMQ, element_type>::ResetNumFilterChecks() {
    for (int i = 0; i < num_layers_; i++) layer_array_[i].num_checks_ = 0;
}

template<template<typename> class BaseAMQ, typename element_type>
void StackedAMQ<BaseAMQ, element_type>::PrintLayerDiagnostics() {
    for (int i = 0; i < num_layers_; i++) {
        printf(
                "Layer %d FPR: %.20f Size:%ld Num element_types:%ld Load "
                "Factor:%f \n",
                i, layer_fprs_[i], layer_array_[i].GetSize(), layer_array_[i].GetNumElements(),
                layer_array_[i].GetLoadFactor());
    }
}
