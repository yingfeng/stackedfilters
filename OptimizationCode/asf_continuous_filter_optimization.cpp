#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>    // std::min
using namespace std;
#include <queue>          // std::queue
#include <chrono> 
using namespace std::chrono;
#include <unordered_set>

#include "utils.h"
#include "workload_generator.h"
#include "ASFContinuousOptimizationObject.h"
#include "ASFUtils.h"
#include "OptimizationRoutines.h"

int ONE_MILLION = 1000000;
int ONE_THOUSAND = 1000;
int TEN_THOUSAND = 10000;
int ONE_HUNDRED_THOUSAND = 100000;

bool optimizeFPRunderSizeConstraint(double sizeBudget, double psi, double NoverP, double bestFPR,
	ASFContinuousOptimizationObject& bestSetup, unsigned int numQueriesToObserve, unsigned int numQueriesAlive) {

  std::queue<ASFContinuousOptimizationObject> myqueue;
  ASFContinuousOptimizationObject startObject(sizeBudget, psi, NoverP, numQueriesToObserve, numQueriesAlive);
  // myqueue.push(OptimizationObject());
  myqueue.push(startObject);
  bool bestFPRset = false;
  while(!myqueue.empty()) {
    ASFContinuousOptimizationObject& currentOptObject = myqueue.front();
    if (currentOptObject.FPRcurr > bestFPR) {
      myqueue.pop();
      continue;
    }

    // analysis here.
    // choose 1 layer
    int a1sizeMax = currentOptObject.constraint;
    ASFContinuousOptimizationObject finalLayerAdded = ASFContinuousOptimizationObject::addFinalLayer(a1sizeMax, currentOptObject);

    if (finalLayerAdded.FPRcurr < bestFPR) {
      bestSetup = finalLayerAdded;
      bestFPR = bestSetup.FPRcurr;
      bestFPRset=true;
    }
    // loop through multi-layer solutions
    // calculate alpha_max for layer 1
    double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * (1 + c) * log(2))), fprForBitsPerEleBF(a1sizeMax) / (0.5 * (1 - currentOptObject.psi)));
    double a1sizeMin = sizeForFprBF(alpha_max);
    if (a1sizeMin <= 0) {
      a1sizeMin = sizeForFprBF(0.5);
    }
    if (currentOptObject.bitsPerElementLayers.size() < 2) {
    	for (double i = a1sizeMin; i <= a1sizeMax; i+= 0.1) {
    		double alpha1 = fprForBitsPerEleBF(i);
        double a2_bound = exp(-1 * (1.0/currentOptObject.NoverP) * (1/alpha1));
        double a2maxBitsPerEle;
        if (a2_bound!=0) {
          a2maxBitsPerEle = sizeForFprBF(a2_bound);
        } else {
          a2maxBitsPerEle = 17.0;
        }
        double sizeLeft = currentOptObject.constraint - i;
        double a2maxBitsPerEle2 = (sizeLeft / (alpha1 * currentOptObject.NoverP)) * pow(log(2), 2);
        a2maxBitsPerEle = min(min(a2maxBitsPerEle, a2maxBitsPerEle2), 17.0);
        if (fprForBitsPerEleBF(a2maxBitsPerEle) > 0.5) {
          continue;
        }
        for (double j = sizeForFprBF(0.5); j <= a2maxBitsPerEle; j+=0.1) {
         ASFContinuousOptimizationObject newOptimizationObject = ASFContinuousOptimizationObject::addFirstAndSecondLayer(i, j, currentOptObject);
         myqueue.push(newOptimizationObject);
       }
     }
   }
   myqueue.pop();
 }
 return bestFPRset;
}

/* used to check the true fpr of optimization method */
double trueFPRCalc(unsigned int numQueriesToObserve, unsigned int numNegativeElements, 
  std::vector<double>& inputSetup, ZipfianDistribution dist,  double proportion, unsigned int numPositiveElements, double constraint) {

  std::vector<double> power_f_values;
  for (unsigned int i = 0; i < numNegativeElements; i++) {
    power_f_values.push_back(pow(1 - getFval(dist.psiVals, i), numQueriesToObserve));
  }
  std::pair<double,double> EpsiENf = calculateEPsiENf(power_f_values, dist.psiVals, numQueriesToObserve, 0);
  double actualSize = inputSetup[0] + (fprForBitsPerEleBF(inputSetup[0]) * (inputSetup[1] * (EpsiENf.second * 1.0) / numPositiveElements)) + (fprForBitsPerEleBF(inputSetup[1]) * inputSetup[2]);
  inputSetup[0] = inputSetup[0] + (constraint - actualSize);
  double fpr = proportion * fprForBitsPerEleBF(inputSetup[0]);
  double nfFPR = (EpsiENf.first * fprForBitsPerEleBF(inputSetup[0]) * fprForBitsPerEleBF(inputSetup[2]));
  double nuFPR = (1 - EpsiENf.first) * fprForBitsPerEleBF(inputSetup[0]);
  fpr += (1 - proportion) * (nfFPR + nuFPR);
  return fpr;
}

ASFContinuousOptimizationObject optimizeASFContinuous(double maxSize, unsigned int numPositiveElements, unsigned int numQueriesAlive, 
  std::vector<double>& psiVals, bool sample) {

  unsigned int numSamples = ONE_HUNDRED_THOUSAND;
  // check one layer filter solution
  double bestFPR = fprForBitsPerEleBF(maxSize);
  //
  ASFContinuousOptimizationObject bestSetup;
  double expectedPsi, expectedNf;
  double unseenValueQueryProportion = 1.0 - psiVals[psiVals.size() - 1];

  unsigned int numQueriesToObserve = 1000;
  unsigned int numQueriesToObserveBest = numQueriesToObserve;
  std::vector<double> power_f_values;
  if (not sample) {
    for (unsigned int i = 0; i < psiVals.size(); i++) {
     power_f_values.push_back(pow(1 - getFval(psiVals,i), numQueriesToObserve));
    }
  }
  while (numQueriesToObserve < numQueriesAlive) {
    if (not sample) {
      std::tie(expectedPsi, expectedNf) = calculateEPsiENf(power_f_values, psiVals, numQueriesToObserve, unseenValueQueryProportion);
    } else {
      std::tie(expectedPsi, expectedNf) = sampledCalculateEPsiENf(psiVals, numQueriesToObserve, unseenValueQueryProportion, numSamples);
    } 
    bool optimizeResults = optimizeFPRunderSizeConstraint(maxSize, expectedPsi, expectedNf / numPositiveElements, bestFPR, bestSetup, numQueriesToObserve, numQueriesAlive);
    if (optimizeResults) {
      numQueriesToObserveBest = numQueriesToObserve;
      bestFPR = bestSetup.FPRcurr;
    }
    numQueriesToObserve = (unsigned int)(numQueriesToObserve * 1.4);
    if (not sample) {
      for (unsigned int i = 0; i < psiVals.size(); i++) {
        power_f_values[i] = pow(power_f_values[i], 1.4);
     }
    }
  }

  if (bestSetup.FPRcurr < fprForBitsPerEleBF(maxSize)) {
    return bestSetup;
  } else {
    return ASFContinuousOptimizationObject::singleLayerFilter(maxSize);
  }
}

int main(int argc, char *argv[]) {

  if (argc < 9) {
    cout << "There should be the following 6 arguments: the minimization (fpr or space),"
    "the constraint (the other of fpr or space), positive elements (in millions), negative elements (in millions), max # negs to take, num queries, zipf" << endl;
    cout << "Ex: ./ASFoptimizer fpr 10.1 1.0 10 2.0 1.0 1.0 false" << endl;
    return 0;
  }

  std::string minimization = argv[1];
  double constraint = std::stod(argv[2]);
  unsigned int numPositiveElements = static_cast<unsigned int>(std::stod(argv[3]) * ONE_MILLION);
  unsigned int numNegativeElements = static_cast<unsigned int>(std::stod(argv[4]) * ONE_MILLION);
  unsigned int maxNumNegativeElements = static_cast<unsigned int>(std::stod(argv[5]) * ONE_MILLION);
  unsigned int numQueriesAlive = static_cast<unsigned int>(std::stod(argv[6]) * ONE_MILLION);
  assert(numNegativeElements > (numPositiveElements / 1000));
  double zipf = std::stod(argv[7]);
  bool sample = stob(argv[8]);
  cout << "zipf is: " << zipf << endl;

  ZipfianDistribution dist = ZipfianDistribution(zipf, numNegativeElements);
  std::vector<double> truncatedPsis;
  for (unsigned int i = 0; i < maxNumNegativeElements; i++) {
    truncatedPsis.push_back(dist.psiVals[i]);
  }

  auto start = high_resolution_clock::now();
  ASFContinuousOptimizationObject returnSetup = optimizeASFContinuous(constraint, numPositiveElements, numQueriesAlive, truncatedPsis, sample);
  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "took " << duration.count() << " milliseconds" << endl; 
  for (auto i: returnSetup.bitsPerElementLayers)
    std::cout << i << ' ';
  cout << endl;
  cout << returnSetup.firstFilterPercentage * numQueriesAlive << endl;

  return 0;
}