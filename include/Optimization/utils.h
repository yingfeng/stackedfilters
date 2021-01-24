#pragma once

#include <sstream>
#include <locale>
#include <iostream>

static constexpr int kCuckooCParam = 2;
static constexpr double kDiscreteLoadFactor = .95;

inline bool stob(std::string s, bool throw_on_error = true)
{
    auto result = false;    // failure to assert is false

    std::istringstream is(s);
    // first try simple integer conversion
    is >> result;

    if (is.fail())
    {
        // simple integer failed; try boolean
        is.clear();
        is >> std::boolalpha >> result;
    }

    if (is.fail() && throw_on_error)
    {
        throw std::invalid_argument(s.append(" is not convertable to bool"));
    }

    return result;
}

inline double sizeInBitsPerElementCF(int fingerPrintSize, double load_factor = .95) {
    return ((double) fingerPrintSize) / load_factor;
}

// will often need to round this function up or down depending on what we want (usually use this for bounds).
inline double nonIntFingerPrintSizeForFprCF(double inputFPR) {
    return (-log2(inputFPR) + 2);
}

inline double fprForFingerPrintSizeCF(int fingerPrintSize) {
    return std::min(1.0, pow(2, kCuckooCParam - fingerPrintSize));
}