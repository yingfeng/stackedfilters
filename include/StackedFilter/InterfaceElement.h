#pragma once

#include <cstddef>
#include "Common.h"

template<typename T>
class InterfaceElement {
public:
    InterfaceElement();

    InterfaceElement(const T &element);

    InterfaceElement(const InterfaceElement &element);

    bool operator<(const InterfaceElement &y) const { return value < y.value; }

    bool operator==(const InterfaceElement &y) const { return value == y.value; }

    T value;
};

template<typename T>
InterfaceElement<T>::InterfaceElement() {
    value = T();
};

template<typename T>
InterfaceElement<T>::InterfaceElement(const T &val) {
    value = val;
};

template<typename T>
InterfaceElement<T>::InterfaceElement(const InterfaceElement &element) {
    value = element.value;
};

class IntElement : public InterfaceElement<int> {
public:
    IntElement() : InterfaceElement<int>() {};

    IntElement(int value) : InterfaceElement<int>(value) {};

    uint32 size() const { return sizeof(int); }

    const char *get_value() const { return (const char *) &value; }
};


class BigIntElement : public InterfaceElement<long int> {
public:
    BigIntElement() : InterfaceElement<long int>() {};

    BigIntElement(long int value) : InterfaceElement<long int>(value) {};

    uint32 size() const { return sizeof(long int); }

    const char *get_value() const { return (const char *) &value; }
};

class StringElement : public InterfaceElement<std::string> {
public:
    StringElement() : InterfaceElement<std::string>() {};

    StringElement(std::string value) : InterfaceElement<std::string>(value) {};

    uint32 size() const { return value.size(); }

    const char *get_value() const { return value.c_str(); }
};

namespace std {
    template<>
    struct hash<StringElement> {
        size_t
        operator()(const StringElement &obj) const {
            return hash<std::string>()(obj.value);
        }
    };

    template<>
    struct hash<IntElement> {
        size_t
        operator()(const IntElement &obj) const {
            return hash<int>()(obj.value);
        }
    };
}