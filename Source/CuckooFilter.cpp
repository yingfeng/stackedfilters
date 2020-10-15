#include <cstring>
#include <cmath>
#include "../Headers/CuckooFilter.h"
#include "../Headers/gqf.h"

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

