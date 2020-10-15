#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <algorithm>    // std::min
using namespace std;
#include <queue>          // std::queue
#include <map>

#include "workload_generator.h"

ZipfianDistribution::ZipfianDistribution(double zipfianParam, int numElements) {
  this->zipfianParam = zipfianParam;
  this->numElements = numElements;
  this->psiVals.reserve(numElements);
  this->computeHarmonic(zipfianParam, static_cast<double>(numElements));  
}

void ZipfianDistribution::computeHarmonic(double zipfianParam, double numElements) {
	double total = 0.0;
	for(double i = 1.0; i <= numElements; i += 1.0) {
		total += 1.0 / (pow(i, zipfianParam));
	}
	double runningTotal = 0.0;
	for (double i = 1.0; i <= numElements; i+= 1.0) {
		runningTotal += 1.0 / (pow(i, zipfianParam));
		psiVals.push_back(runningTotal / total);
	}
	this->totalHarmonicVal = total;
}

double ZipfianDistribution::getPsiVal(unsigned int index) {
	return psiVals[index];
}

double ZipfianDistribution::getFval(unsigned int index) {
  if (index == 0) {
    return psiVals[0];
  } else {
    double fVal = psiVals[index] - psiVals[index-1];
    assert(fVal != 0);
    return fVal;
  }
}
