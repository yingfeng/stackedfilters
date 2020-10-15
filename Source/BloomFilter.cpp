#include "../Headers/BloomFilter.h"

template<typename element_type>
BloomFilter<element_type>::BloomFilter(size_t filter_size, int num_hashes,
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
BloomFilter<element_type>::~BloomFilter() {}

_GLIBCXX17_INLINE
template<typename element_type>
bool BloomFilter<element_type>::LookupElement(const element_type element) {
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
void BloomFilter<element_type>::InsertElement(const element_type element) {
    this->num_elements_++;
    size_t hash1 = getHash1(element);
    size_t hash2 = getHash2(element);
    for (unsigned int i = 0; i < num_hashes_; i++) {
        filter_[getNthHash(hash1, hash2, i)] = true;
    }
}

_GLIBCXX17_INLINE
template<typename element_type>
size_t BloomFilter<element_type>::getHash1(const element_type x) {
    return CityHash64WithSeed(x.get_value(), x.size(), 123456789 + seed_);
}

_GLIBCXX17_INLINE
template<typename element_type>
size_t BloomFilter<element_type>::getHash2(const element_type x) {
    return CityHash64WithSeed(x.get_value(), x.size(), 987654321 + seed_);
}

_GLIBCXX17_INLINE
template<typename element_type>
// Double hashing strategy recommended in Mitzenmacher Paper.
size_t BloomFilter<element_type>::getNthHash(const size_t hash1, const size_t hash2,
                                             const size_t hash_num) {
    return (hash1 + hash_num * hash2 + hash_num * hash_num * hash_num) % filter_size_;
}

template<typename element_type>
double BloomFilter<element_type>::GetLoadFactor() {
    size_t load = 0;
    for (size_t i = 0; i < filter_size_; i++)
        if (filter_[i] == true) load++;
    return (double) load / (double) filter_size_;
}
