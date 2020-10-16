#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <algorithm>    // std::min
#include "ASFUtils.h"
#include "utils.h"

using namespace std;


class ASFDiscreteOptimizationObject {
public:
    // field variables
    double constraint;
    double psi;
    double NoverP;
    double FPRcurr;
    double FPRFirstFilter;
    double proportion;
    double firstFilterPercentage;
    std::vector<int> fingerprints;

    ASFDiscreteOptimizationObject() {
    }

    ASFDiscreteOptimizationObject(double constraint, double psi, double NoverP,
                                  unsigned int numQueriesToObserve, unsigned int totalNumQueries) {
        this->constraint = constraint;
        this->psi = psi;
        this->NoverP = NoverP;
        this->firstFilterPercentage = (double(numQueriesToObserve)) / totalNumQueries;
        assert(this->firstFilterPercentage != 0);
        this->proportion = 1.0;
        this->FPRcurr = 0.0;
    }

    ASFDiscreteOptimizationObject(double constraint, double psi, double NoverP, unsigned int numQueriesToObserve,
                                  unsigned int totalNumQueries, double FPRcurr, double proportion,
                                  std::vector<int> fingerprints) {
        this->constraint = constraint;
        this->psi = psi;
        this->NoverP = NoverP;
        this->firstFilterPercentage = (double(numQueriesToObserve)) / totalNumQueries;
        assert(this->firstFilterPercentage != 0);
        this->FPRcurr = FPRcurr;
        this->proportion = proportion;
        this->fingerprints = fingerprints;
    }

    static ASFDiscreteOptimizationObject singleLayerFilter(int f1) {
        ASFDiscreteOptimizationObject returnObject;
        returnObject.firstFilterPercentage = 1.0;
        returnObject.FPRcurr = fprForFingerPrintSizeCF(f1);
        returnObject.fingerprints.push_back(f1);
        return returnObject;
    }

    static ASFDiscreteOptimizationObject
    addFirstAndSecondLayer(int f1, int f2, const ASFDiscreteOptimizationObject &currentOptObject) {
        ASFDiscreteOptimizationObject returnObject;
        double alpha1 = fprForFingerPrintSizeCF(f1);
        double alpha2 = fprForFingerPrintSizeCF(f2);
        returnObject.firstFilterPercentage = currentOptObject.firstFilterPercentage;
        returnObject.constraint = (currentOptObject.constraint - f1 - currentOptObject.NoverP * alpha1 * f2) / alpha2;
        returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
        returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
        returnObject.proportion =
                currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
        returnObject.FPRcurr = (currentOptObject.firstFilterPercentage * alpha1) +
                               (1.0 - currentOptObject.firstFilterPercentage) * (1 - currentOptObject.psi) * alpha1 *
                               (1 - alpha2);
        returnObject.FPRFirstFilter = alpha1;
        returnObject.fingerprints = currentOptObject.fingerprints;
        returnObject.fingerprints.push_back(f1);
        returnObject.fingerprints.push_back(f2);
        return returnObject;
    }

    static ASFDiscreteOptimizationObject addFinalLayer(int f1, const ASFDiscreteOptimizationObject &currentOptObject) {
        ASFDiscreteOptimizationObject returnObject;
        double alpha1 = fprForFingerPrintSizeCF(f1);
        returnObject.firstFilterPercentage = currentOptObject.firstFilterPercentage;
        returnObject.constraint = currentOptObject.constraint - f1;
        returnObject.psi = currentOptObject.psi;
        returnObject.NoverP = currentOptObject.NoverP * alpha1;
        returnObject.proportion = currentOptObject.proportion * alpha1;
        if (currentOptObject.fingerprints.size() != 0) {
            returnObject.FPRcurr = currentOptObject.FPRcurr +
                                   (1.0 - currentOptObject.firstFilterPercentage) * (1 - currentOptObject.psi) * alpha1;
        } else {
            returnObject.FPRcurr = alpha1;
        }
        returnObject.fingerprints = currentOptObject.fingerprints;
        returnObject.fingerprints.push_back(f1);
        return returnObject;
    }

/* to go deeper beyond one or two layers, use the following */
    static ASFDiscreteOptimizationObject
    createNewOptimizationObject(int f1, int f2, const ASFDiscreteOptimizationObject &currentOptObject) {
        ASFDiscreteOptimizationObject returnObject;
        double alpha1 = fprForFingerPrintSizeCF(f1);
        double alpha2 = fprForFingerPrintSizeCF(f2);
        returnObject.constraint = (currentOptObject.constraint - f1 - currentOptObject.NoverP * alpha1 * f2) / alpha2;
        returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
        returnObject.firstFilterPercentage = currentOptObject.firstFilterPercentage;
        returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
        returnObject.proportion =
                currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
        returnObject.FPRcurr = currentOptObject.FPRcurr +
                               (1.0 - currentOptObject.firstFilterPercentage) * (1 - currentOptObject.psi) * alpha1 *
                               (1 - alpha2);
        returnObject.fingerprints = currentOptObject.fingerprints;
        returnObject.fingerprints.push_back(f1);
        returnObject.fingerprints.push_back(f2);
        return returnObject;
    }
};
