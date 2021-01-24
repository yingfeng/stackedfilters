//
// Created by kylebd99 on 9/9/19.
//

#ifndef STACKEDALTERNATINGAMQS_CUCKOOFILTERLAYER_H
#define STACKEDALTERNATINGAMQS_CUCKOOFILTERLAYER_H

#include <cstddef>
#include <vector>
#include <cstring>
#include <cmath>
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"
#include "cuckoofilter.h"

using cuckoofilter::CuckooFilter;

template<typename element_type>
class CuckooFilterLayer : public InterfaceAMQ<CuckooFilterLayer, element_type> {
private:
public:
    CuckooFilter<2> *filter_2_;
    CuckooFilter<3> *filter_3_;
    CuckooFilter<4> *filter_4_;
    CuckooFilter<5> *filter_5_;
    CuckooFilter<6> *filter_6_;
    CuckooFilter<7> *filter_7_;
    CuckooFilter<8> *filter_8_;
    CuckooFilter<9> *filter_9_;
    CuckooFilter<10> *filter_10_;
    CuckooFilter<11> *filter_11_;
    CuckooFilter<12> *filter_12_;
    CuckooFilter<13> *filter_13_;
    CuckooFilter<14> *filter_14_;
    CuckooFilter<15> *filter_15_;
    CuckooFilter<16> *filter_16_;

    unsigned int bits_per_key_;
    uint64 max_elements_;

    CuckooFilterLayer(size_t filter_size, uint32 bits_per_key_, uint32 seed);

    CuckooFilterLayer() {
    };

    ~CuckooFilterLayer();

    uint64 getHash(element_type element);

    _GLIBCXX17_INLINE
    bool LookupElement(element_type element) override;

    _GLIBCXX17_INLINE
    void InsertElement(element_type element) override;

    double GetLoadFactor();

    static size_t SizeFunctionImplementation(double fpr,
                                             size_t num_expected_elements) {
        uint32 num_remainder_bits = 16;
        const std::vector<uint32> possible_bits = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        for (int i = 0; i < possible_bits.size(); i++) {
            double temp_fpr = pow(1.0 / 2, possible_bits[i]) * 8;
            if (temp_fpr <= fpr) {
                num_remainder_bits = possible_bits[i];
                break;
            }
        }
        // uint64 nslots = pow(2, ceil(log2(num_expected_elements * 1.065)));
        return (num_remainder_bits) * cuckoofilter::upperpower2(num_expected_elements * 1.06);
    }
};



using cuckoofilter::CuckooFilter;

template<typename element_type>
CuckooFilterLayer<element_type>::CuckooFilterLayer(const uint64 filter_size, const uint32 bits_per_key,
                                                   const uint32 seed) {
    const std::vector<uint32> possible_bits = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    for (int i = 0; i < possible_bits.size(); i++) {
        if (bits_per_key <= possible_bits[i]) {
            bits_per_key_ = possible_bits[i];
            break;
        }
    }
    uint64 max_slots = cuckoofilter::upperpower2(filter_size / bits_per_key_) + 4;
    max_elements_ = max_slots;
    this->total_size_ = max_slots * bits_per_key_;
    if (bits_per_key_ == 2) filter_2_ = new CuckooFilter<2>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 3) filter_3_ = new CuckooFilter<3>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 4) filter_4_ = new CuckooFilter<4>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 5) filter_5_ = new CuckooFilter<5>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 6) filter_6_ = new CuckooFilter<6>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 7) filter_7_ = new CuckooFilter<7>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 8) filter_8_ = new CuckooFilter<8>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 9) filter_9_ = new CuckooFilter<9>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 10) filter_10_ = new CuckooFilter<10>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 11) filter_11_ = new CuckooFilter<11>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 12) filter_12_ = new CuckooFilter<12>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 13) filter_13_ = new CuckooFilter<13>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 14) filter_14_ = new CuckooFilter<14>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 15) filter_15_ = new CuckooFilter<15>(max_slots, rand() % 50000000);
    if (bits_per_key_ == 16) filter_16_ = new CuckooFilter<16>(max_slots, rand() % 50000000);
}

template<typename element_type>
CuckooFilterLayer<element_type>::~CuckooFilterLayer() {
};

_GLIBCXX17_INLINE
template<typename element_type>
bool CuckooFilterLayer<element_type>::LookupElement(const element_type element) {
    //this->num_checks_++;
    //uint64 hash = getHash(element);
    if (bits_per_key_ == 2) return filter_2_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 3) return filter_3_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 4) return filter_4_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 5) return filter_5_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 6) return filter_6_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 7) return filter_7_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 8) return filter_8_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 9) return filter_9_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 10) return filter_10_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 11) return filter_11_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 12) return filter_12_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 13) return filter_13_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 14) return filter_14_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 15) return filter_15_->Contain(element.value) == cuckoofilter::Status::Ok;
    if (bits_per_key_ == 16) return filter_16_->Contain(element.value) == cuckoofilter::Status::Ok;
}

_GLIBCXX17_INLINE
template<typename element_type>
void CuckooFilterLayer<element_type>::InsertElement(const element_type element) {
    this->num_elements_++;
    //uint64 hash = getHash(element);
    cuckoofilter::Status ret = cuckoofilter::Status::NotSupported;
    if (bits_per_key_ == 2) ret = filter_2_->Add(element.value);
    if (bits_per_key_ == 3) ret = filter_3_->Add(element.value);
    if (bits_per_key_ == 4) ret = filter_4_->Add(element.value);
    if (bits_per_key_ == 5) ret = filter_5_->Add(element.value);
    if (bits_per_key_ == 6) ret = filter_6_->Add(element.value);
    if (bits_per_key_ == 7) ret = filter_7_->Add(element.value);
    if (bits_per_key_ == 8) ret = filter_8_->Add(element.value);
    if (bits_per_key_ == 9) ret = filter_9_->Add(element.value);
    if (bits_per_key_ == 10) ret = filter_10_->Add(element.value);
    if (bits_per_key_ == 11) ret = filter_11_->Add(element.value);
    if (bits_per_key_ == 12) ret = filter_12_->Add(element.value);
    if (bits_per_key_ == 13) ret = filter_13_->Add(element.value);
    if (bits_per_key_ == 14) ret = filter_14_->Add(element.value);
    if (bits_per_key_ == 15) ret = filter_15_->Add(element.value);
    if (bits_per_key_ == 16) ret = filter_16_->Add(element.value);
    if (ret == cuckoofilter::Status::NotEnoughSpace) {
        printf("CQF Ran Out Of Space!\n");
    }
}

template<typename element_type>
uint64 CuckooFilterLayer<element_type>::getHash(element_type x) {
    return x.value;
}

template<typename element_type>
double CuckooFilterLayer<element_type>::GetLoadFactor() {
    return (double) this->num_elements_ / max_elements_;
}



template
class CuckooFilterLayer<IntElement>;

#endif STACKEDALTERNATINGAMQS_CUCKOOFILTERLAYER_H
