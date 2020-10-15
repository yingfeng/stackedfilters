#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <algorithm>    // std::min
using namespace std;

double fprForBitsPerEleBF(double bitsPerEle) {
  return min(1.0, pow(2, -1 * log(2) * bitsPerEle));
}

double sizeForFprBF(double inputFPR) {
  return -log(inputFPR) * (1.0 / pow(log(2), 2));
}

class ASFContinuousOptimizationObject {
  public:
    // field variables
    double constraint;
    double psi;
    double NoverP;
    double FPRcurr;
    double FPRFirstFilter;
    double proportion;
    double firstFilterPercentage;
    std::vector<double> bitsPerElementLayers;
    // constructors
    ASFContinuousOptimizationObject();
    ASFContinuousOptimizationObject(double constraint, double psi, double NoverP, unsigned int numQueriesToObserve, 
      unsigned int totalNumQueries, double FPRcurr, double proportion, std::vector<double> bpeLayers);
    ASFContinuousOptimizationObject(double constraint, double psi, double NoverP, unsigned int numQueriesToObserve, unsigned int totalNumQueries);
    // methods for creating from an input object
    static ASFContinuousOptimizationObject addFinalLayer(double bitsEle1, ASFContinuousOptimizationObject currentOptObject);
    static ASFContinuousOptimizationObject createNewOptimizationObject(double bitsEle1, double bitsEle2, ASFContinuousOptimizationObject currentOptObject);
    static ASFContinuousOptimizationObject addFirstAndSecondLayer(double bitsEle1, double bitsEle2, ASFContinuousOptimizationObject currentOptObject);
    static ASFContinuousOptimizationObject singleLayerFilter(double bitsPerEle);
};

ASFContinuousOptimizationObject::ASFContinuousOptimizationObject() {
}

ASFContinuousOptimizationObject::ASFContinuousOptimizationObject(double constraint, double psi, double NoverP, 
  unsigned int numQueriesToObserve, unsigned int totalNumQueries) {
  this->constraint = constraint;
  this->psi = psi;
  this->NoverP = NoverP;
  this->firstFilterPercentage = (double(numQueriesToObserve)) / totalNumQueries;
  assert(this->firstFilterPercentage != 0);
  this->proportion = 1.0;
  this->FPRcurr = 0.0;
}

ASFContinuousOptimizationObject::ASFContinuousOptimizationObject(double constraint, double psi, double NoverP, unsigned int numQueriesToObserve, 
      unsigned int totalNumQueries, double FPRcurr, double proportion, std::vector<double> bitsPerElementLayers) {
  this->constraint = constraint;
  this->psi = psi;
  this->NoverP = NoverP;
  this->firstFilterPercentage = (double(numQueriesToObserve)) / totalNumQueries;
  assert(this->firstFilterPercentage != 0);
  this->FPRcurr = FPRcurr;
  this->proportion = proportion;
  this->bitsPerElementLayers = bitsPerElementLayers;
}

ASFContinuousOptimizationObject ASFContinuousOptimizationObject::singleLayerFilter(double bitsPerEle) {
  ASFContinuousOptimizationObject returnObject;
  returnObject.firstFilterPercentage = 1.0;
  returnObject.FPRcurr = fprForBitsPerEleBF(bitsPerEle);
  returnObject.bitsPerElementLayers.push_back(bitsPerEle);
  return returnObject;
}

ASFContinuousOptimizationObject ASFContinuousOptimizationObject::addFirstAndSecondLayer(double bitsEle1, double bitsEle2, ASFContinuousOptimizationObject currentOptObject) {
  ASFContinuousOptimizationObject returnObject;
  double alpha1 = fprForBitsPerEleBF(bitsEle1);
  double alpha2 = fprForBitsPerEleBF(bitsEle2);
  returnObject.firstFilterPercentage = currentOptObject.firstFilterPercentage;
  returnObject.constraint = (currentOptObject.constraint - bitsEle1 - currentOptObject.NoverP * alpha1 * bitsEle2) / alpha2;
  returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
  returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
  returnObject.proportion = currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
  returnObject.FPRcurr = (currentOptObject.firstFilterPercentage * alpha1) + (1.0 - currentOptObject.firstFilterPercentage) * (1 - currentOptObject.psi) * alpha1 * (1 - alpha2);
  returnObject.FPRFirstFilter = alpha1;
  returnObject.bitsPerElementLayers = currentOptObject.bitsPerElementLayers;
  returnObject.bitsPerElementLayers.push_back(bitsEle1);
  returnObject.bitsPerElementLayers.push_back(bitsEle2);
  return returnObject;
}

ASFContinuousOptimizationObject ASFContinuousOptimizationObject::addFinalLayer(double bitsEle1, ASFContinuousOptimizationObject currentOptObject) {
  ASFContinuousOptimizationObject returnObject;
  double alpha1 = fprForBitsPerEleBF(bitsEle1);
  returnObject.firstFilterPercentage = currentOptObject.firstFilterPercentage;
  returnObject.constraint = currentOptObject.constraint - bitsEle1;
  returnObject.psi = currentOptObject.psi;
  returnObject.NoverP = currentOptObject.NoverP * alpha1;
  returnObject.proportion = currentOptObject.proportion * alpha1;
  returnObject.FPRFirstFilter = currentOptObject.FPRFirstFilter;
  if (currentOptObject.bitsPerElementLayers.size() != 0) {
    returnObject.FPRcurr = currentOptObject.FPRcurr + (1.0 - currentOptObject.firstFilterPercentage) * (1 - currentOptObject.psi) * alpha1;
  } else {
    returnObject.FPRcurr = alpha1;
  }
  returnObject.bitsPerElementLayers = currentOptObject.bitsPerElementLayers;
  returnObject.bitsPerElementLayers.push_back(bitsEle1);
  return returnObject;
}

/* to go deeper beyond one or two layers, use the following */
ASFContinuousOptimizationObject ASFContinuousOptimizationObject::createNewOptimizationObject(double bitsEle1, double bitsEle2, ASFContinuousOptimizationObject currentOptObject) {
  ASFContinuousOptimizationObject returnObject;
  double alpha1 = fprForBitsPerEleBF(bitsEle1);
  double alpha2 = fprForBitsPerEleBF(bitsEle2);
  returnObject.constraint = (currentOptObject.constraint - bitsEle1 - currentOptObject.NoverP * alpha1 * bitsEle2) / alpha2;
  returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
  returnObject.firstFilterPercentage = currentOptObject.firstFilterPercentage;
  returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
  returnObject.proportion = currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
  returnObject.FPRcurr = currentOptObject.FPRcurr + (1.0 - currentOptObject.firstFilterPercentage) * (1 - currentOptObject.psi) * alpha1 * (1 - alpha2);
  returnObject.bitsPerElementLayers = currentOptObject.bitsPerElementLayers;
  returnObject.bitsPerElementLayers.push_back(bitsEle1);
  returnObject.bitsPerElementLayers.push_back(bitsEle2);
  return returnObject;
}
