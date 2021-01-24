#pragma once

#include <iostream>
#include <vector>
using namespace std;
#include <map>

class ZipfianDistribution {
  public:
    double zipfianParam;
    int numElements;
    double totalHarmonicVal;
    std::vector<double> psiVals;
    ZipfianDistribution(double zipfianParam, int numElements);
    double getPsiVal(unsigned int index);
    double getFval(unsigned int index);

  private:
  	void computeHarmonic(double zipfianParam, double numElements);
}; 