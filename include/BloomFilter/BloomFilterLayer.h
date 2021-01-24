#pragma once

#include <cstddef>
#include <vector>
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"

template<typename element_type>
class BloomFilterLayer : public InterfaceAMQ<BloomFilterLayer, element_type> {
private:
public:
    std::vector<bool> filter_;
    unsigned int num_hashes_;
    size_t filter_size_;
    int seed_;

    BloomFilterLayer(size_t filter_size, int num_hashes, int seed);

    BloomFilterLayer() : BloomFilterLayer(0, 0, 0) {};

    ~BloomFilterLayer();

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


template<typename element_type>
BloomFilterLayer<element_type>::BloomFilterLayer(size_t filter_size, int num_hashes,
                                                 int seed) {
    seed_ = seed;
    num_hashes_ = num_hashes;
    this->total_size_ = filter_size;
    filter_size_ = filter_size;
    filter_.resize(filter_size, false);
    if (filter_size <= 2) {
        filter_size_ = 2;
        filter_.resize(filter_size_, true);
    }
}

template<typename element_type>
BloomFilterLayer<element_type>::~BloomFilterLayer() {}

_GLIBCXX17_INLINE
template<typename element_type>
bool BloomFilterLayer<element_type>::LookupElement(const element_type element) {
    size_t hash1 = getHash1(element);
    size_t hash2 = getHash2(element);
    for (unsigned int i = 0; i < num_hashes_; i++) {
        if (filter_[getNthHash(hash1, hash2, i)] == false) {
            return false;
        }
    }
    return true;
}

_GLIBCXX17_INLINE
template<typename element_type>
void BloomFilterLayer<element_type>::InsertElement(const element_type element) {
    this->num_elements_++;
    size_t hash1 = getHash1(element);
    size_t hash2 = getHash2(element);
    for (unsigned int i = 0; i < num_hashes_; i++) {
        filter_[getNthHash(hash1, hash2, i)] = true;
    }
}

_GLIBCXX17_INLINE
template<typename element_type>
size_t BloomFilterLayer<element_type>::getHash1(const element_type x) {
    return CityHash64WithSeed(x.get_value(), x.size(), 123456789 + seed_);
}

_GLIBCXX17_INLINE
template<typename element_type>
size_t BloomFilterLayer<element_type>::getHash2(const element_type x) {
    return CityHash64WithSeed(x.get_value(), x.size(), 987654321 + seed_);
}

_GLIBCXX17_INLINE
template<typename element_type>
// Double hashing strategy recommended in Mitzenmacher Paper.
size_t BloomFilterLayer<element_type>::getNthHash(const size_t hash1, const size_t hash2,
                                                  const size_t hash_num) {
    return (hash1 + hash_num * hash2 + hash_num * hash_num * hash_num) % filter_size_;
}

template<typename element_type>
double BloomFilterLayer<element_type>::GetLoadFactor() {
    size_t load = 0;
    for (size_t i = 0; i < filter_size_; i++)
        if (filter_[i] == true) load++;
    return (double) load / (double) filter_size_;
}

template
class BloomFilterLayer<IntElement>;

template
class BloomFilterLayer<StringElement>;

template
class BloomFilterLayer<BigIntElement>;
