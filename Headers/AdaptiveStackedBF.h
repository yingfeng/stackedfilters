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
#include <iterator>
#include <experimental/algorithm>


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
    std::vector<BloomFilter<element_type>> layer_array_;
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
            size_t total_size, size_t total_queries, const std::vector<double> &pmf, size_t sample_estimate_size);

    void InitAdaptiveStackedBF(const std::vector<element_type> &positives);

    ~AdaptiveStackedBF();

    bool LookupElement(element_type element);

    void InsertPositive(element_type element);

    FilterStatus DeclareFalsePositiveAndCheckStatus(element_type element);

    void BuildThirdLayer(const std::vector<element_type> &positives);

    void RebuildFilter(const std::vector<element_type> &positives = std::vector<element_type>(),
                       const std::vector<double> &pmf = std::vector<double>());

    void DeleteElement(element_type element);

    size_t GetSize();

    void CalculateLayerFPRsAndFPCapacity(const std::vector<double> &pmf);

    size_t NumFilterChecks();

    void ResetNumFilterChecks();

    void PrintLayerDiagnostics();
};

template
class AdaptiveStackedBF<IntElement>;

template
class AdaptiveStackedBF<StringElement>;

template
class AdaptiveStackedBF<BigIntElement>;