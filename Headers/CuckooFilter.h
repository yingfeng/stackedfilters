//
// Created by kylebd99 on 9/9/19.
//

#ifndef STACKEDALTERNATINGAMQS_CUCKOOFILTER_H
#define STACKEDALTERNATINGAMQS_CUCKOOFILTER_H

#include <cstddef>
#include <vector>
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

template
class CuckooFilterLayer<IntElement>;

#endif STACKEDALTERNATINGAMQS_CUCKOOFILTER_H
