#pragma once

#include <iostream>
#include <vector>

#include "utils.h"
#include "ASFDiscreteOptimizationObject.h"
#include "ASFContinuousOptimizationObject.h"
#include "ASFUtils.h"

std::vector<double> optimizeStackedFilterBloom(double maxSize, unsigned int numPositiveElements, double epsilonSlack, std::vector<double>& psiPerNegativeElement);

ASFDiscreteOptimizationObject optimizeASFDiscrete(double maxSize, unsigned int numPositiveElements, unsigned int numQueriesAlive, 
  std::vector<double>& psiVals, bool sample);

ASFContinuousOptimizationObject optimizeASFContinuous(double maxSize, unsigned int numPositiveElements, unsigned int numQueriesAlive, 
  std::vector<double>& psiVals, bool sample);

std::vector<int> optimizeDiscreteStackedFilter(double maxSize, unsigned int numPositiveElements, double epsilonSlack, std::vector<double>& psiPerNegativeElement, double load_factor);