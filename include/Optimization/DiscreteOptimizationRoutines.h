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

#include "workload_generator.h"

//int c = 2;
//double load_factor = 0.95;

class OptimizationObject {
public:
    double constraint;
    double psi;
    double NoverP;
    double slack;
    double FPRcurr;
    double proportion;
    int countBelow;
    std::vector<int> fingerprints;


    OptimizationObject() {
    }

    OptimizationObject(double constraint, double psi, double NoverP, double slack) {
        this->constraint = constraint;
        this->psi = psi;
        this->NoverP = NoverP;
        this->slack = slack;
        this->proportion = 1.0;
        this->FPRcurr = 0.0;
    }

    OptimizationObject(double constraint, double psi, double NoverP, double slack,
                       double FPRcurr, double proportion, std::vector<int> fingerprints) {
        this->constraint = constraint;
        this->psi = psi;
        this->NoverP = NoverP;
        this->slack = slack;
        this->FPRcurr = FPRcurr;
        this->proportion = proportion;
        this->fingerprints = fingerprints;
        this->countBelow = 0;
    }

};

inline OptimizationObject createNewOptimizationObject(int f1, int f2, OptimizationObject currentOptObject) {
    OptimizationObject returnObject;
    double alpha1 = fprForFingerPrintSizeCF(f1);
    double alpha2 = fprForFingerPrintSizeCF(f2);
    returnObject.constraint = (currentOptObject.constraint - f1 - currentOptObject.NoverP * alpha1 * f2) / alpha2;
    returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
    returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
    returnObject.slack = currentOptObject.slack;
    returnObject.proportion =
            currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
    returnObject.FPRcurr = currentOptObject.FPRcurr + (1 - currentOptObject.psi) * alpha1 * (1 - alpha2);
    returnObject.fingerprints = currentOptObject.fingerprints;
    returnObject.fingerprints.push_back(f1);
    returnObject.fingerprints.push_back(f2);
    return returnObject;
}

inline void printFingerprints(OptimizationObject currentOptObject) {
    cout << "inputFingerprints ";
    for (auto i: currentOptObject.fingerprints)
        std::cout << i << ' ';
    std::cout << endl;
}

inline double calculateSize(std::vector<int> fingerprints, double NoverP) {
    double posSize = 0.0;
    double negSize = 0.0;
    for (unsigned int i = 0; i < fingerprints.size(); i++) {
        if (i % 2 == 0) {
            double proportion = 1.0;
            for (unsigned int j = 0; j < i / 2; j++) {
                proportion = proportion * fprForFingerPrintSizeCF(fingerprints[(2 * j) + 1]);
            }
            posSize += proportion * fingerprints[i];
        } else {
            double proportion = 1.0;
            for (unsigned int j = 0; j < (i + 1) / 2; j++) {
                proportion = proportion * fprForFingerPrintSizeCF(fingerprints[2 * j]);
            }
            negSize += NoverP * proportion * fingerprints[i];
        }
    }
    double totalSize = posSize + negSize;
    return totalSize;
}

inline double calculateEFPR(std::vector<int> fingerprints, double psi) {
    double proportion = 1.0;
    double FPRcurr = 0.0;
    double psiTrack = psi;
    for (unsigned int i = 0; i < fingerprints.size() / 2; i++) {
        double alpha1 = fprForFingerPrintSizeCF(fingerprints[i]);
        double alpha2 = fprForFingerPrintSizeCF(fingerprints[i + 1]);
        proportion = proportion * alpha1 * (psi + (1 - psi) * alpha2);
        FPRcurr = FPRcurr + (1 - psiTrack) * alpha1 * (1 - alpha2);
        psiTrack = psiTrack / (psiTrack + ((1 - psiTrack) * alpha2));
    }
    if (fingerprints.size() > 0) {
        FPRcurr += proportion * fprForFingerPrintSizeCF(fingerprints[fingerprints.size() - 1]);
    } else {
        FPRcurr += proportion;
    }
    return FPRcurr;
}

//pair<OptimizationObject, int>
inline pair<int, bool>
optimizeFPRunderSizeConstraint(double sizeBudget, double psi, double NoverP, double epsilonSlack, double bestFPR,
                               OptimizationObject &bestSetup) {

    std::queue<OptimizationObject> myqueue;
    OptimizationObject startObject(sizeBudget, psi, NoverP, epsilonSlack);
    myqueue.push(startObject);
    long long int totalSearched = 0;
    bool bestFPRset = false;
    while (!myqueue.empty()) {
        OptimizationObject &currentOptObject = myqueue.front();
        totalSearched += 1;
        if (currentOptObject.FPRcurr > bestFPR) {
            myqueue.pop();
            continue;
        }

        // analysis here.
        // choose 1 layer
        int a1FingerPrintMax = floor(currentOptObject.constraint);
        if (a1FingerPrintMax <= kCuckooCParam) {
            myqueue.pop();
            continue;
        }
        double addedFPR = currentOptObject.proportion * fprForFingerPrintSizeCF(a1FingerPrintMax);
        if (totalSearched % 10000000 == 0) {
            cout << totalSearched / 1000000 << endl;
        }
        if (currentOptObject.FPRcurr + addedFPR < bestFPR) {
            bestFPR = currentOptObject.FPRcurr + addedFPR;
            bestSetup = currentOptObject;
            bestSetup.fingerprints.push_back(a1FingerPrintMax);
            bestSetup.proportion = 0.0;
            bestSetup.FPRcurr += addedFPR;
            bestFPRset = true;
        }

        // check if a single layer solution puts us within epsilon of best possible using the current stack
        // if all queries on negatives were caught, difference would be less than epsilon. Stop here.
        if (addedFPR < epsilonSlack) {
            myqueue.pop();
            continue;
        }
        // loop through multi-layer solutions
        // calculate alpha_max for layer 1
        double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * (1 + kCuckooCParam) * log(2))),
                               fprForFingerPrintSizeCF(a1FingerPrintMax) / (0.5 * (1 - currentOptObject.psi)));
        int a1FingerprintMin = ceil(nonIntFingerPrintSizeForFprCF(alpha_max));
        //cout << "min and max for this layer f1: " << a1FingerprintMin << ", " << a1FingerPrintMax << endl;
        for (int i = a1FingerprintMin; i <= a1FingerPrintMax; i++) {
            double alpha1 = fprForFingerPrintSizeCF(i);
            // see math derivation. c comes from the fact that best integer value may be past when dSize/ dAlpha1 < 0.
            double a2_bound = pow(2,
                                  (-1 * (1.0 / currentOptObject.NoverP) * (1 / alpha1) * (1 / log(2))) + kCuckooCParam);
            int a2maxFingerprintSize;
            if (a2_bound != 0) {
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
            for (int j = kCuckooCParam + 1; j <= a2maxFingerprintSize; j++) {
                OptimizationObject newOptimizationObject = createNewOptimizationObject(i, j, currentOptObject);
                myqueue.push(newOptimizationObject);
            }
        }

        myqueue.pop();
    }
    return std::pair<int, bool>(totalSearched, bestFPRset);
}

inline std::pair<uint64_t, std::vector<int>>
optimizeDiscreteStackedFilter(const double maxSize, const unsigned int numPositiveElements, const double epsilonSlack,
                              const std::vector<double> &psiPerNegativeElement, const double load_factor) {
    // remove load factor. Makes all subsequent optimization much easier
    double constraint = maxSize * load_factor;

    // calculate single layer FPR
    int maxFingerprintSize = floor(constraint);
    double bestFPR = fprForFingerPrintSizeCF(maxFingerprintSize);
    double originalFPR = bestFPR;

    // do discrete search over all fingerprint sizes
    double lastPsi = 0;
    OptimizationObject bestSetup;
    int numSearched = 0;
    double NoverPused = 0.0;
    uint64_t sizeOfNf = 0;

    // sweep over N_samp
    for (unsigned int i = 1; i <= psiPerNegativeElement.size(); i += std::max<size_t>(50, psiPerNegativeElement.size()/1000)) {
        double newPsi = psiPerNegativeElement[i - 1];
        double slackFromN = (newPsi - lastPsi) * min(originalFPR / (1 - newPsi), 1.0);
        if (slackFromN > epsilonSlack / 2) {
            double NoverP = static_cast<double>(i) / static_cast<double>(numPositiveElements);
            // inner optimization
            pair<int, bool> optimizeResults = optimizeFPRunderSizeConstraint(constraint, newPsi, NoverP,
                                                                             epsilonSlack / 2, bestFPR, bestSetup);
            lastPsi = newPsi;
            numSearched += optimizeResults.first;
            if (optimizeResults.second) {
                NoverPused = NoverP;
            }
        }
    }
    sizeOfNf = bestSetup.NoverP * numPositiveElements;
    if (bestSetup.FPRcurr < fprForFingerPrintSizeCF(maxFingerprintSize)) {
        std::vector<int> returnSetup = bestSetup.fingerprints;
        return {sizeOfNf, returnSetup};
    } else {
        return {sizeOfNf, std::vector<int>{maxFingerprintSize}};
    }
}