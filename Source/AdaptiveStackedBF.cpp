#include "../Headers/AdaptiveStackedBF.h"

// Calculate the proper layer FPRs and sample size then hand off to the constructor that takes in
// layer fprs.
template<typename element_type>
AdaptiveStackedBF<element_type>::AdaptiveStackedBF(
        const std::vector<element_type> &positives,
        const size_t total_size, const size_t total_queries, const std::vector<double> &pmf,
        const size_t sample_estimate_size) {
    sample_estimate_size_ = sample_estimate_size;
    num_positive_ = positives.size();
    total_size_ = total_size;
    total_queries_ = total_queries;
    queries_left_ = total_queries;
    eng_ = std::mt19937_64(rd_());

    CalculateLayerFPRsAndFPCapacity(pmf);
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
                                                    const std::vector<double> &pmf) {
    // Clear the second and third filters
    fully_built_ = false;
    threshold_fpr_ = -1;
    num_false_positives_inserted_ = 0;
    layer_array_[1].filter_.resize(0);
    layer_array_.resize(2);
    // If a full rebuild is not requested, the second layer is simply reserved again.
    if (positives.empty() || pmf.empty()) {
        uint64_t second_layer_size = BloomFilter<IntElement>::SizeFunction(layer_fprs_[1], false_positives_capacity_);
        layer_array_[1].filter_.resize(second_layer_size, false);
        current_status_ = OPERATIONAL;
        return;
    }
    // For a full rebuild, the optimization is re-run, and the first layer is rebuilt.
    CalculateLayerFPRsAndFPCapacity(pmf);
    InitAdaptiveStackedBF(positives);
    current_status_ = OPERATIONAL;
}

template<typename element_type>
void AdaptiveStackedBF<element_type>::DeleteElement(element_type element) {}


/* Optimization Methods */

static double rank_to_size(const size_t *collection_size, const double rank, const size_t sample_estimate_size) {
    long index = (size_t) (rank * sample_estimate_size);
    if (index >= sample_estimate_size) {
        return 1;
    }
    if (index < 0) {
        return 0;
    }
    return collection_size[index];
}

struct SizeParameters {
    size_t *collection_size;
    size_t sample_estimate_size;
    size_t total_size;
    size_t num_positive;
    uint32 num_layers;
};

static double SizeFunctionVaried(unsigned num_variables, const double *rank_and_layer_fprs,
                                 double *grad, void *size_params_ptr) {
    auto *params = (SizeParameters *) size_params_ptr;
    size_t total_size = params->total_size * .995;
    size_t num_positive = params->num_positive;
    size_t sample_estimate_size = params->sample_estimate_size;
    size_t num_negative = rank_to_size(params->collection_size, rank_and_layer_fprs[0], sample_estimate_size);
    const double *layer_fprs = rank_and_layer_fprs + 1;
    double size = 0;
    size += BloomFilter<IntElement>::SizeFunction(layer_fprs[0], num_positive);
    size += BloomFilter<IntElement>::SizeFunction(layer_fprs[1], layer_fprs[0] * num_negative);
    size += BloomFilter<IntElement>::SizeFunction(layer_fprs[2], layer_fprs[1] * num_positive);
    return size - total_size;
}

static double rank_to_psi(const double *collection_psi, const double rank, const size_t sample_estimate_size) {
    size_t index = (size_t) (rank * sample_estimate_size);
    if (index >= sample_estimate_size) {
        return 1;
    }
    if (index < 0) {
        return 0;
    }
    return collection_psi[index];
}


struct FprParameters {
    size_t sample_estimate_size;
    double penalty_coef;
    double *collection_psi;
    uint32 num_layers;
};

static double FprFunctionVaried(unsigned num_variables, const double *rank_and_layer_fprs,
                                double *grad, void *fpr_params_ptr) {
    auto *params = (FprParameters *) fpr_params_ptr;
    const double penalty_coef = params->penalty_coef;
    uint32 num_layers = params->num_layers;
    size_t sample_estimate_size = params->sample_estimate_size;
    double cold_query_proportion = rank_and_layer_fprs[0];
    double psi = rank_to_psi(params->collection_psi, cold_query_proportion, sample_estimate_size);
    const double *layer_fprs = rank_and_layer_fprs + 1;
    double known_fpr = layer_fprs[0] * layer_fprs[2];
    double unknown_fpr_side = layer_fprs[0] * (1 - layer_fprs[1]);
    double unknown_fpr_end = layer_fprs[0] * layer_fprs[1] * layer_fprs[2];
    // Penalty Function For Number of Hashes
    int num_hashes = 0;
    for (uint32 i = 0; i < num_layers; i++) num_hashes += std::max((int) (round(-log(layer_fprs[i]) / log(2))), 1);
    double cold_fpr = layer_fprs[0];
    double warm_fpr = psi * known_fpr +
                      (1 - psi) * (unknown_fpr_side + unknown_fpr_end);
    double total_fpr = cold_query_proportion * cold_fpr + (1 - cold_query_proportion) * warm_fpr;
    return total_fpr *
           (1 + num_hashes * penalty_coef);
}

std::pair<double *, size_t *> get_collection_estimate(const std::vector<double> &pmf, const size_t sample_estimate_size,
                                                      const size_t max_sample_size) {
    auto *collection_psi = new double[sample_estimate_size];
    auto *collection_size = new size_t[sample_estimate_size];
    double collection_size_step = (double) max_sample_size / sample_estimate_size;

    constexpr uint64_t kPMFSampleSize = 100000;
    std::vector<double> pmf_sample(kPMFSampleSize);
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0, pmf.size());
    for (uint64_t sample_idx = 0; sample_idx < kPMFSampleSize; sample_idx++)
        pmf_sample.push_back(pmf[distribution(generator)]);

    for (size_t collection_estimate_index = 1;
         collection_estimate_index < sample_estimate_size; collection_estimate_index++) {
        collection_psi[collection_estimate_index] = 0;
        double temp_size = 0;
        for (const auto &p : pmf_sample) {
            collection_psi[collection_estimate_index] +=
                    p * pow(1 - p, collection_estimate_index * collection_size_step);
            temp_size +=
                    1 - pow(1 - p, collection_estimate_index * collection_size_step);
        }
        collection_psi[collection_estimate_index] = 1 - collection_psi[collection_estimate_index] * pmf.size() /
                                                        static_cast<double>(kPMFSampleSize);
        collection_size[collection_estimate_index] = temp_size * pmf.size() / static_cast<double>(kPMFSampleSize);
    }
    return {collection_psi, collection_size};
}


// Use NLOPT to calculate layer fprs and the number of unique false positives to be included.
template<typename element_type>
void AdaptiveStackedBF<element_type>::CalculateLayerFPRsAndFPCapacity(const std::vector<double> &pmf) {
    double *collection_psi;
    size_t *collection_size;
    std::tie(collection_psi, collection_size) = get_collection_estimate(pmf, sample_estimate_size_,
                                                                        queries_left_);
    constexpr uint32 num_variables = num_layers_ + 1;
    std::vector<double> rank_and_layer_fprs(num_variables, 0.0001);

    FprParameters fpr_params{};
    fpr_params.sample_estimate_size = sample_estimate_size_;
    fpr_params.penalty_coef = penalty_coef_;
    fpr_params.collection_psi = collection_psi;
    fpr_params.num_layers = num_layers_;
    SizeParameters size_params{};
    size_params.collection_size = collection_size;
    size_params.total_size = total_size_;
    size_params.num_positive = num_positive_;
    size_params.sample_estimate_size = sample_estimate_size_;
    size_params.num_layers = num_layers_;

    // Calculate the FPR of an equal-fpr stacked filter.
    auto lower_bounds = (double *) calloc(num_variables, sizeof(double));
    for (uint32 i = 0; i < num_variables; i++) lower_bounds[i] = 0.00000000000000000001;
    auto upper_bounds = (double *) calloc((num_variables), sizeof(double));
    for (uint32 i = 0; i < num_variables; i++) upper_bounds[i] = 1;
    double max_rank = 1;
    upper_bounds[0] = max_rank;
    nlopt_opt local_fpr_opt = nlopt_create(NLOPT_GN_ISRES, num_variables);
    nlopt_set_lower_bounds(local_fpr_opt, lower_bounds);
    nlopt_set_upper_bounds(local_fpr_opt, upper_bounds);
    nlopt_set_maxtime(local_fpr_opt, .25);
    nlopt_add_inequality_constraint(
            local_fpr_opt, SizeFunctionVaried,
            &size_params, total_size_ * .0005);
    nlopt_set_min_objective(local_fpr_opt, FprFunctionVaried,
                            &fpr_params);
    double variable_fpr_fpr = 1;
    nlopt_result local_ret_status =
            nlopt_optimize(local_fpr_opt, rank_and_layer_fprs.data(), &variable_fpr_fpr);
    if (local_ret_status == -4)
        printf("ERROR!!!  Roundoff Errors Reached in Local Optimization\n");
    else if (local_ret_status < 0)
        printf("ERROR!!! General Error in Local Optimization: %d\n", local_ret_status);
    free(upper_bounds);
    free(lower_bounds);

    layer_fprs_ = std::vector<double>(rank_and_layer_fprs.begin() + 1, rank_and_layer_fprs.end());

    // Set the threshold FPR to be halfway between the cold and warm FPR.
    const double psi = rank_to_psi(collection_psi, rank_and_layer_fprs[0], sample_estimate_size_);
    target_fpr_ = layer_fprs_[0] *
                  ((1 - psi) * ((1 - layer_fprs_[1]) + layer_fprs_[1] * layer_fprs_[2]) + psi * layer_fprs_[2]);
    const double single_layer_fpr = exp(-static_cast<double>(total_size_) / num_positive_ * log(2) * log(2));
    threshold_fpr_ = -1;

    // Adjust for the fact that only elements which flip at least one bit will count towards the capacity
    size_t negatives_size = rank_to_size(collection_size, rank_and_layer_fprs[0], sample_estimate_size_);
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
