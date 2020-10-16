#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <algorithm>    // std::min
using namespace std;
#include <queue>          // std::queue
#include <chrono>
using namespace std::chrono;


#include "ASFUtils.h"
#include "utils.h"
#include "ASFDiscreteOptimizationObject.h"
#include "workload_generator.h"

inline pair<int, bool> optimizeFPRunderSizeConstraint(double sizeBudget, double psi, double NoverP, double bestFPR,
                                               ASFDiscreteOptimizationObject& bestSetup, unsigned int numQueriesToObserve, unsigned int numQueriesAlive) {

    std::queue<ASFDiscreteOptimizationObject> myqueue;
    ASFDiscreteOptimizationObject startObject(sizeBudget, psi, NoverP, numQueriesToObserve, numQueriesAlive);
    // myqueue.push(OptimizationObject());
    myqueue.push(startObject);
    long long int totalSearched = 0;
    bool bestFPRset = false;
    while(!myqueue.empty()) {
        ASFDiscreteOptimizationObject& currentOptObject = myqueue.front();
        if (currentOptObject.FPRcurr > bestFPR) {
            myqueue.pop();
            continue;
        }

        int a1FingerPrintMax = floor(currentOptObject.constraint);
        if (a1FingerPrintMax <= kCuckooCParam) {
            myqueue.pop();
            continue;
        }
        ASFDiscreteOptimizationObject finalLayerAdded = ASFDiscreteOptimizationObject::addFinalLayer(a1FingerPrintMax, currentOptObject);

        if (finalLayerAdded.FPRcurr < bestFPR) {
            bestSetup = finalLayerAdded;
            bestFPR = bestSetup.FPRcurr;
            bestFPRset=true;
        }
        // loop through multi-layer solutions
        // calculate alpha_max for layer 1
        double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * (1 + kCuckooCParam) * log(2))), fprForFingerPrintSizeCF(a1FingerPrintMax) / (0.5 * (1 - currentOptObject.psi)));
        int a1FingerprintMin = ceil(nonIntFingerPrintSizeForFprCF(alpha_max));
        if (currentOptObject.fingerprints.size() < 2) {
            for (int i = a1FingerprintMin; i <= a1FingerPrintMax; i++) {
                double alpha1 = fprForFingerPrintSizeCF(i);
                // see math derivation. c comes from the fact that best integer value may be past when dSize/ dAlpha1 < 0.
                double a2_bound = pow(2, (-1 * (1.0/currentOptObject.NoverP) * (1/alpha1) * (1/log(2))) + kCuckooCParam);
                int a2maxFingerprintSize;
                if (a2_bound!=0) {
                    a2maxFingerprintSize = ceil(nonIntFingerPrintSizeForFprCF(a2_bound));
                } else {
                    a2maxFingerprintSize = 20;
                }
                double sizeLeft = currentOptObject.constraint - i;
                int a2maxFingerprintSize2 = floor(sizeLeft / (alpha1 * currentOptObject.NoverP));
                a2maxFingerprintSize = min(min(a2maxFingerprintSize, a2maxFingerprintSize2), 17);
                if (fprForFingerPrintSizeCF(a2maxFingerprintSize) > 0.5) {
                    continue;
                }
                for (int j = kCuckooCParam+1; j <= a2maxFingerprintSize; j++) {
                    ASFDiscreteOptimizationObject newOptimizationObject = ASFDiscreteOptimizationObject::addFirstAndSecondLayer(i, j, currentOptObject);
                    myqueue.push(newOptimizationObject);
                }
            }
        }
        myqueue.pop();
    }
    return std::pair<int,bool>(totalSearched, bestFPRset);
}
inline ASFDiscreteOptimizationObject optimizeASFDiscrete(double maxSize, unsigned int numPositiveElements, unsigned int numQueriesAlive,
                                                  std::vector<double>& psiVals, bool sample) {

    // remove off load_factor. easier for optimization.
    double constraint = maxSize * kDiscreteLoadFactor;
    double unseenValueQueryProportion = 1.0 - psiVals[psiVals.size()-1];
    unsigned int numSamples = 100000;

    // 1) calculate single layer FPR. Use this to get epsilon slack for best FPR allowed.
    int maxFingerprintSize = floor(constraint);
    double bestFPR = fprForFingerPrintSizeCF(maxFingerprintSize);

    ASFDiscreteOptimizationObject bestSetup;
    double expectedPsi, expectedNf;

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
        pair<int, bool> optimizeResults = optimizeFPRunderSizeConstraint(constraint, expectedPsi, expectedNf / numPositiveElements, bestFPR, bestSetup, numQueriesToObserve, numQueriesAlive);
        if (optimizeResults.second) {
            numQueriesToObserveBest = numQueriesToObserve;
            bestFPR = bestSetup.FPRcurr;
        }
        numQueriesToObserve = (unsigned int)(numQueriesToObserve * 1.4);
        if (not sample) {
            for (unsigned int i = 0; i < power_f_values.size(); i++) {
                power_f_values[i] = pow(power_f_values[i], 1.4);
            }
        }
    }

    std::vector<int> returnSetup = bestSetup.fingerprints;
    if (bestSetup.FPRcurr < fprForFingerPrintSizeCF(maxFingerprintSize)) {
        return bestSetup;
    } else {
        return ASFDiscreteOptimizationObject::singleLayerFilter(maxFingerprintSize);
    }
}