//
// Created by kylebd99 on 9/9/19.
//

#ifndef STACKEDALTERNATINGAMQS_CQFILTER_H
#define STACKEDALTERNATINGAMQS_CQFILTER_H

#include <cstddef>
#include <vector>
#include "InterfaceAMQ.h"
#include "InterfaceElement.h"
#include "gqf.h"
#include "gqf_int.h"

template<typename element_type>
class CQFilter : public InterfaceAMQ<CQFilter, element_type> {
private:
public:
    QF filter_;
    unsigned int num_hash_bits_;
    unsigned int num_remainder_bits_;
    int seed_;

    CQFilter(size_t filter_size, int num_remainder_bits, int seed);

    CQFilter() : CQFilter(0, 0, 0) {};

    ~CQFilter();

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

template
class CQFilter<IntElement>;

#endif //STACKEDALTERNATINGAMQS_CQFILTER_H
