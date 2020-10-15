#pragma once

#include <cstddef>
#include <vector>
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"

template<typename element_type>
class BloomFilter : public InterfaceAMQ<BloomFilter, element_type> {
private:
public:
    std::vector<bool> filter_;
    unsigned int num_hashes_;
    size_t filter_size_;
    int seed_;

    BloomFilter(size_t filter_size, int num_hashes, int seed);

    BloomFilter() : BloomFilter(0, 0, 0) {};

    ~BloomFilter();

    size_t getHash1(const element_type element);

    size_t getHash2(const element_type element);

    size_t getNthHash(const size_t hash1, const size_t hash2, const size_t hash_num);

    bool LookupElement(const element_type element);

    void InsertElement(const element_type element);

    double GetLoadFactor();

    static size_t SizeFunctionImplementation(double fpr,
                                             size_t num_expected_elements) {
        size_t num_hashes = std::max<size_t>(ceil(-std::log2(fpr)), 1);
        size_t theoretical_size = -1. / (std::pow(1. - std::pow(fpr, 1. / num_hashes),
                                                  1. / (num_hashes * num_expected_elements)) - 1.);
        return theoretical_size; // Adjusted to keep load factors <=.5
    }
};

template
class BloomFilter<IntElement>;

template
class BloomFilter<StringElement>;

template
class BloomFilter<BigIntElement>;
