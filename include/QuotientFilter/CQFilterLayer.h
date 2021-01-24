//
// Created by kylebd99 on 9/9/19.
//

#ifndef STACKEDALTERNATINGAMQS_CQFILTERLAYER_H
#define STACKEDALTERNATINGAMQS_CQFILTERLAYER_H

#include <cstddef>
#include <vector>
#include <cstring>
#include <cmath>
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"
#include "gqf.h"
#include "gqf_int.h"

template<typename element_type>
class CQFilterLayer : public InterfaceAMQ<CQFilterLayer, element_type> {
private:
public:
    QF filter_;
    unsigned int num_hash_bits_;
    unsigned int num_remainder_bits_;
    int seed_;

    CQFilterLayer(size_t filter_size, int num_remainder_bits, int seed);

    CQFilterLayer() : CQFilterLayer(0, 0, 0) {};

    ~CQFilterLayer();

    uint64 getHash(element_type element);

    _GLIBCXX17_INLINE
    bool LookupElement(const element_type element) override;

    _GLIBCXX17_INLINE
    void InsertElement(const element_type element) override;

    double GetLoadFactor();

    static size_t SizeFunctionImplementation(double fpr,
                                             size_t num_expected_elements) {
        uint32 num_remainder_bits = std::max<uint32>(ceil(-log2(fpr)), 2);
        // uint64 nslots = pow(2, ceil(log2(num_expected_elements * 1.065)));
        return (num_remainder_bits + 2.125) * num_expected_elements * 1.06;
    }
};


template<typename element_type>
CQFilterLayer<element_type>::CQFilterLayer(const uint64 filter_size, int num_remainder_bits,
                                 int seed) {
    seed_ = seed;
    num_remainder_bits_ = std::max<uint32>(num_remainder_bits, 2);
    this->total_size_ = filter_size;
    uint64 nslots = std::max<uint64>(filter_size / (num_remainder_bits + 2.125), 1);
    qf_malloc(&filter_, nslots, num_remainder_bits_ + log2(nslots), 0, QF_HASH_DEFAULT, seed, false);
    qf_set_auto_resize(&filter_, false);
}

template<typename element_type>
CQFilterLayer<element_type>::~CQFilterLayer() {
}

_GLIBCXX17_INLINE
template<typename element_type>
bool CQFilterLayer<element_type>::LookupElement(const element_type element) {
    //this->num_checks_++;
    //uint64 hash = getHash(element);
    return qf_count_key_value(&filter_, element.value, 0, QF_NO_LOCK);
}

_GLIBCXX17_INLINE
template<typename element_type>
void CQFilterLayer<element_type>::InsertElement(const element_type element) {
    this->num_elements_++;
    //uint64 hash = getHash(element);
    int ret = qf_set_count(&filter_, element.value, 0, 1, QF_NO_LOCK);
    if (ret == -1) {
        printf("CQF Ran Out Of Space!\n");
    }
}

template<typename element_type>
uint64 CQFilterLayer<element_type>::getHash(element_type x) {
    return x.value;
}

template<typename element_type>
double CQFilterLayer<element_type>::GetLoadFactor() {
    return (double) qf_get_num_occupied_slots(&filter_) / qf_get_nslots(&filter_);
}

template
class CQFilterLayer<IntElement>;


#endif //STACKEDALTERNATINGAMQS_CQFILTERLAYER_H
