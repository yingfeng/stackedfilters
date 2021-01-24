#pragma once

#include <cstddef>
#include "Common.h"

// AMQs used in the stacked filter will be derived from this following
// CRTP https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
template<template<typename element_type> class T, typename element_type>
class InterfaceAMQ {
protected:
    double fpr_;
    size_t total_size_;

public:
    size_t num_checks_ = 0;
    size_t num_elements_ = 0;

    virtual ~InterfaceAMQ() {};

    virtual void InsertElement(element_type element) = 0;

    virtual bool LookupElement(element_type element) = 0;

    virtual void DeleteElement(element_type element) {};

    static size_t SizeFunction(double fpr, size_t num_expected_elements) {
        return T<element_type>::SizeFunctionImplementation(fpr,
                                                           num_expected_elements);
    };

    size_t GetSize() { return total_size_; };

    size_t GetNumElements() { return num_elements_; };

    double GetFPR() { return fpr_; };
};
