#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm> 
 
double getFval(std::vector<double>& psiVals, unsigned int i) {
  if (i == 0) {
    return psiVals[0];
  } else {
    double fVal = psiVals[i] - psiVals[i-1];
    assert(fVal != 0);
    return fVal;
  }
}

std::pair<double,double> sampledCalculateEPsiENf(std::vector<double>& psiVals, 
  unsigned int numQueriesToObserve, double unseenValueQueryProportion, unsigned int numSamples) {
  // amount to overcount sampled values
  double inverseProportionSampled = (1.0 * psiVals.size()) / numSamples;
  // generate rng
  std::default_random_engine generator;
  generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<uint64_t> distribution(0, psiVals.size()-1);
  // expected number of unseen values. 
  double unseenExpected = unseenValueQueryProportion * numQueriesToObserve;
  double expectedNf = unseenExpected;
  double expectedPsi = 0;
  for (unsigned int i = 0; i < numSamples; i++) {
      unsigned int sample_idx = distribution(generator);
      double power_f_value = pow(1 - getFval(psiVals, sample_idx), numQueriesToObserve);
      expectedPsi += getFval(psiVals, sample_idx) * power_f_value;
      expectedNf += inverseProportionSampled * (1 - power_f_value);
  }
  expectedPsi = 1 - unseenValueQueryProportion - (inverseProportionSampled * expectedPsi);
  return std::make_pair(expectedPsi, expectedNf);
}

double calculateExpectedPsi(std::vector<double> &power_f_values, std::vector<double>& psiVals) {
	double fullPsi = psiVals[psiVals.size() - 1];
  assert(psiVals.size() == power_f_values.size());
	for (unsigned int i = 0; i < psiVals.size(); i++) {
		fullPsi -= (getFval(psiVals, i) * power_f_values[i]);
	}
	return fullPsi;
}

double calculateExpectedNf(std::vector<double> &power_f_values, unsigned int numQueriesToObserve, double unseenValueQueryProportion) {
	double unseenExpected = unseenValueQueryProportion * numQueriesToObserve;
  double returnResult = unseenExpected + power_f_values.size();
  for (unsigned int i = 0; i < power_f_values.size(); i++) {
    returnResult -= power_f_values[i];
  }
  return returnResult;
}

std::pair<double, double> calculateEPsiENf(std::vector<double> &power_f_values, std::vector<double>& psiVals,
  unsigned int numQueriesToObserve, double unseenValueQueryProportion) {
  double expectedPsi = calculateExpectedPsi(power_f_values, psiVals);
  double expectedNumNegs = calculateExpectedNf(power_f_values, numQueriesToObserve, unseenValueQueryProportion);
  return std::make_pair(expectedPsi, expectedNumNegs);
}