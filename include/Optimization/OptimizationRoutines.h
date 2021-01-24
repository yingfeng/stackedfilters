#pragma once

#include <iostream>
#include <vector>

#include "ASFDiscreteOptimizationRoutines.h"
#include "ASFContinuousOptimizationRoutines.h"
#include "BloomOptimizationRoutines.h"
#include "DiscreteOptimizationRoutines.h"

/*
std::pair<uint64_t, std::vector<double>> optimizeStackedFilterBloom(double maxSize, unsigned int numPositiveElements, double epsilonSlack, const std::vector<double>& psiPerNegativeElement);

ASFDiscreteOptimizationObject optimizeASFDiscrete(double maxSize, unsigned int numPositiveElements, unsigned int numQueriesAlive, 
  std::vector<double>& psiVals, bool sample);

ASFContinuousOptimizationObject optimizeASFContinuous(double maxSize, unsigned int numPositiveElements, unsigned int numQueriesAlive, 
  std::vector<double>& psiVals, bool sample);

std::pair<uint64_t, std::vector<int>>  optimizeDiscreteStackedFilter(double maxSize, unsigned int numPositiveElements, double epsilonSlack, const std::vector<double>& psiPerNegativeElement, double load_factor);
 */