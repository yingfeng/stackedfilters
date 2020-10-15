#pragma once

#include <sstream>
#include <locale>
#include <iostream>

bool stob(std::string s, bool throw_on_error = true)
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

