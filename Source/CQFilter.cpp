#include <cstring>
#include <cmath>
#include "../Headers/CQFilter.h"
#include "../Headers/gqf.h"

template<typename element_type>
CQFilter<element_type>::CQFilter(const uint64 filter_size, int num_remainder_bits,
                                 int seed) {
    seed_ = seed;
    num_remainder_bits_ = std::max<uint32>(num_remainder_bits, 2);
    this->total_size_ = filter_size;
    uint64 nslots = std::max<uint64>(filter_size / (num_remainder_bits + 2.125), 1);
    qf_malloc(&filter_, nslots, num_remainder_bits_ + log2(nslots), 0, QF_HASH_DEFAULT, seed, false);
    qf_set_auto_resize(&filter_, false);
}

template<typename element_type>
CQFilter<element_type>::~CQFilter() {
}

_GLIBCXX17_INLINE
template<typename element_type>
bool CQFilter<element_type>::LookupElement(const element_type element) {
    //this->num_checks_++;
    //uint64 hash = getHash(element);
    return qf_count_key_value(&filter_, element.value, 0, QF_NO_LOCK);
}

_GLIBCXX17_INLINE
template<typename element_type>
void CQFilter<element_type>::InsertElement(const element_type element) {
    this->num_elements_++;
    //uint64 hash = getHash(element);
    int ret = qf_set_count(&filter_, element.value, 0, 1, QF_NO_LOCK);
    if (ret == -1) {
        printf("CQF Ran Out Of Space!\n");
    }
}

template<typename element_type>
uint64 CQFilter<element_type>::getHash(element_type x) {
    return x.value;
}

template<typename element_type>
double CQFilter<element_type>::GetLoadFactor() {
    return (double) qf_get_num_occupied_slots(&filter_) / qf_get_nslots(&filter_);
}

