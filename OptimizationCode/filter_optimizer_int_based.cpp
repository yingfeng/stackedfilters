#include <iostream>
#include <vector>
#include <string>
#include <cstdarg>
#include <cmath>
#include <cstring>
#include <algorithm>    // std::min
using namespace std;
#include <queue>          // std::queue

#include "workload_generator.h"

int c = 2;

//WRITE COUNTING OF EACH DEPTH.

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
    OptimizationObject();
    OptimizationObject(double constraint, double psi, double NoverP, double slack, double FPRcurr, double proportion, std::vector<int> fingerprints);
    OptimizationObject(double constraint, double psi, double NoverP, double slack);
};

OptimizationObject::OptimizationObject() {
}

OptimizationObject::OptimizationObject(double constraint, double psi, double NoverP, double slack) {
  this->constraint = constraint;
  this->psi = psi;
  this->NoverP = NoverP;
  this->slack = slack;
  this->proportion = 1.0;
}

OptimizationObject::OptimizationObject(double constraint, double psi, double NoverP, double slack, 
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

double sizeInBitsPerElementCF(int fingerPrintSize) {
  return ((double) fingerPrintSize) / 0.95;
}

double fprForFingerPrintSizeCF(int fingerPrintSize) {
  return min(1.0, pow(2, c - fingerPrintSize));
}

// will often need to round this function up or down depending on what we want (usually use this for bounds). 
double nonIntFingerPrintSizeForFprCF(double inputFPR) {
  return (-log2(inputFPR) + c);
}

OptimizationObject createNewOptimizationObject(int f1, int f2, OptimizationObject currentOptObject) {
  OptimizationObject returnObject;
  double alpha1 = fprForFingerPrintSizeCF(f1);
  double alpha2 = fprForFingerPrintSizeCF(f2);
  returnObject.constraint = (currentOptObject.constraint - sizeInBitsPerElementCF(f1) - currentOptObject.NoverP * alpha1 * sizeInBitsPerElementCF(f2)) / alpha2;
  returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
  returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
  returnObject.slack = currentOptObject.slack;
  returnObject.proportion = currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
  returnObject.FPRcurr = currentOptObject.FPRcurr + (1 - currentOptObject.psi) * alpha1 * (1 - alpha2);
  returnObject.fingerprints = currentOptObject.fingerprints;
  returnObject.fingerprints.push_back(f1);
  returnObject.fingerprints.push_back(f2);
  return returnObject;
}

double getTotalLogVariation(int f1, int f2,OptimizationObject currentOptObject, bool print) {
  double si = currentOptObject.constraint * 0.95;
  double numerator = ((1 - pow(2, (-1 * f2) + c)) * si) - f1 - (currentOptObject.NoverP * pow(2, (-1 * f1) + c) * f2);
  double variation = (f1 - c) + ((1.0 / pow(2,(-1 * f2) + c)) * numerator);
  double new_si = (si - f1 - (currentOptObject.NoverP * pow(2, (-1 * f1) + c) * f2)) * (1.0 / pow(2,(-1 * f2) + c));
  if (new_si < (1 + c)) {
    return 25;
  }
  if (print and currentOptObject.fingerprints.size() < 2) {
    //if (variation < 0) {
      std::cout << "input Fingerprints ";
      for (auto i: currentOptObject.fingerprints) {
        std::cout << i << ' '; 
      }
      cout << "f1: " << f1 << " f2: " << f2;
      cout << " NoverP: " << currentOptObject.NoverP << " si: " << si << "new si: " << new_si << " variation: " << variation;
      cout << endl;
    //}
  }
  return variation;
}

void printFingerprints(OptimizationObject currentOptObject) {
  cout << "inputFingerprints ";
  for (auto i: currentOptObject.fingerprints)
    std::cout << i << ' ';
  std::cout << endl;
}

OptimizationObject optimizeFPRunderSizeConstraintDepthFirst(OptimizationObject currentOptObject) {
  std::vector<int> inputFingerprints = currentOptObject.fingerprints;
  OptimizationObject returnObject = currentOptObject;
  //std:: cout << " size: " << currentOptObject.constraint << endl;
  int a1FingerPrintMax = floor(currentOptObject.constraint * 0.95);
  double addedFPR = currentOptObject.proportion * fprForFingerPrintSizeCF(a1FingerPrintMax);
  returnObject.countBelow = 1;
  // check if done
  if (addedFPR < currentOptObject.slack) {  
    returnObject.FPRcurr = currentOptObject.FPRcurr + addedFPR;
    returnObject.fingerprints.push_back(a1FingerPrintMax);
    return returnObject;
  }

  double bestFPR = currentOptObject.FPRcurr + addedFPR;
  std::vector<int> bestFingerprints;

  double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * (1 + c) * log(2))), fprForFingerPrintSizeCF(a1FingerPrintMax) / (0.5 * (1 - currentOptObject.psi))); 
  int a1FingerprintMin = floor(nonIntFingerPrintSizeForFprCF(alpha_max));
  //int a1FingerprintMin = ceil(nonIntFingerPrintSizeForFprCF(alpha_max));
  a1FingerprintMin = max(c+1, a1FingerprintMin);
  //cout << "min and max for this layer f1: " << a1FingerprintMin << ", " << a1FingerPrintMax << endl;
  for (int i = a1FingerprintMin; i <= a1FingerPrintMax; i++) {
    double alpha1 = fprForFingerPrintSizeCF(i);
    // see math derivation. c - 1 comes from the fact that best integer value may be past when dSize/ dAlpha1 < 0.  
    double a2_bound = pow(2, (-1 * (1.0/currentOptObject.NoverP) * (1/alpha1) * (1/log(2))) + c);
    int a2maxFingerprintSize;
    if (a2_bound!=0) {
      a2maxFingerprintSize = ceil(nonIntFingerPrintSizeForFprCF(a2_bound));
    } else {
      a2maxFingerprintSize = 20;
    }
    //cout << "for f1, a1 bound: " << i << ", " << alpha1 << " the a2,f2 bound is " << a2_bound << "," << a2maxFingerprintSize << endl;
    double sizeLeft = currentOptObject.constraint - (i / 0.95);
    int a2maxFingerprintSize2 = floor((sizeLeft / (alpha1 * currentOptObject.NoverP)) * 0.95);
    a2maxFingerprintSize = min(min(a2maxFingerprintSize, a2maxFingerprintSize2), 17);
    //cout << "for f1, a1 bound: " << i << ", " << alpha1 << " the eventual bound is " << a2maxFingerprintSize << endl;
    if (fprForFingerPrintSizeCF(a2maxFingerprintSize) > 0.5) {
      continue;
    }
    for (int j = c+1; j <= a2maxFingerprintSize; j++) {
      OptimizationObject newOptimizationObject = createNewOptimizationObject(i, j, currentOptObject);
      //getTotalLogVariation(i,j,currentOptObject, true);
      //cout << "f1, f2: " << i << j << endl;
      newOptimizationObject = optimizeFPRunderSizeConstraintDepthFirst(newOptimizationObject);
      returnObject.countBelow += newOptimizationObject.countBelow;
      if (newOptimizationObject.FPRcurr < bestFPR) {
        bestFPR = newOptimizationObject.FPRcurr;
        bestFingerprints = newOptimizationObject.fingerprints;
      }
    }
  }

  returnObject.FPRcurr = bestFPR;
  returnObject.fingerprints = bestFingerprints;

  //cout << "inputFingerprints ";
  //for (auto i: inputFingerprints)
  //  std::cout << i << ' ';
  //cout << ", # setups is " << returnObject.countBelow << endl;
  //cout << "FPR of best setup: " << bestFPR << endl;

  return returnObject;

}

std::vector<int> initializeBuckets() {
  std::vector<int> buckets;
  for (int i = 0; i < 22; i++) {
    buckets.push_back(0);
  }
  return buckets;
}

void incrementBuckets(std::vector<int>& incrementVec, double totalVariation) {
  int intVariation = floor(totalVariation);
  if (intVariation < 0) {
    incrementVec[21] += 1;
  } else if (intVariation > 20) {
    incrementVec[20] += 1;
  } else {
    incrementVec[intVariation] += 1;
  }
}

void bucketPrint(std::vector<int>& totalVariationBuckets) {
  for (unsigned int i = 0; i < totalVariationBuckets.size(); i++) {
    cout << "floor of variation: " << i << ", #: " << totalVariationBuckets[i] << endl;
  }
}
 
std::vector<int> optimizeFPRunderSizeConstraint(double sizeBudget, double psi, double NoverP, double epsilonSlack) {

  std::queue<OptimizationObject> myqueue;
  OptimizationObject startObject(sizeBudget,psi, NoverP, epsilonSlack);
  OptimizationObject bestSetup = startObject;
  // myqueue.push(OptimizationObject());
  myqueue.push(startObject);
  long long int totalSearched = 0;
  // should be overwritten immediately. 
  double bestFPR = 1.1;
  std::vector<int> buckets = initializeBuckets();
  while(!myqueue.empty()) {
    OptimizationObject& currentOptObject = myqueue.front();
    totalSearched += 1;
    //printFingerprints(currentOptObject);
    /*if (currentOptObject.FPRcurr > bestFPR) {
      myqueue.pop();
      continue;
    } */

    // analysis here.
    // choose 1 layer
    int a1FingerPrintMax = floor(currentOptObject.constraint * 0.95);
    double addedFPR = currentOptObject.proportion * fprForFingerPrintSizeCF(a1FingerPrintMax);
    if (totalSearched % 1000000 == 0) {
      cout << totalSearched / 1000000 << endl;
    }
    if (currentOptObject.FPRcurr + addedFPR < bestFPR) {
      //cout << "sanity check" << endl;
      bestFPR = currentOptObject.FPRcurr + addedFPR;
      bestSetup = currentOptObject;
      bestSetup.fingerprints.push_back(a1FingerPrintMax);
    }

    // WE COULD STILL MAKE THIS FASTER BY CHECKING IF CURRENT FPR > BEST FPR

    // check if a single layer solution puts us within epsilon of best possible using the current stack
    // if all queries on negatives were caught, difference would be less than epsilon. Stop here. 
    
    if (addedFPR < epsilonSlack) {
      myqueue.pop();
      continue;
    }
    //if (currentOptObject.proportion < epsilonSlack) {
    //  myqueue.pop();
    //  continue;
    //}
    
    // loop through multi-layer solutions
    // calculate alpha_max for layer 1
    double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * (1 + c) * log(2))), fprForFingerPrintSizeCF(a1FingerPrintMax) / (0.5 * (1 - currentOptObject.psi)));
    int a1FingerprintMin = floor(nonIntFingerPrintSizeForFprCF(alpha_max));
    if (a1FingerprintMin <= 0) {
      a1FingerprintMin = c+1;
    }
    //cout << "a1FingerprintMin: " << a1FingerprintMin << endl;

    //int a1FingerprintMin = ceil(nonIntFingerPrintSizeForFprCF(alpha_max));

    for (int i = a1FingerprintMin; i <= a1FingerPrintMax; i++) {
      //cout << "i: " << i << endl;
      double alpha1 = fprForFingerPrintSizeCF(i);
      // see math derivation. c - 1 comes from the fact that best integer value may be past when dSize/ dAlpha1 < 0.  
      double a2_bound = pow(2, (-1 * (1.0/currentOptObject.NoverP) * (1/alpha1) * (1/log(2))) + c);
      int a2maxFingerprintSize;
      if (a2_bound!=0) {
        a2maxFingerprintSize = ceil(nonIntFingerPrintSizeForFprCF(a2_bound));
      } else {
        a2maxFingerprintSize = 20;
      }
      //cout << "for f1, a1 bound: " << i << ", " << alpha1 << " the a2,f2 bound is " << a2_bound << "," << a2maxFingerprintSize << endl;
      double sizeLeft = currentOptObject.constraint - (i / 0.95);
      int a2maxFingerprintSize2 = floor((sizeLeft / (alpha1 * currentOptObject.NoverP)) * 0.95);
      a2maxFingerprintSize = min(min(a2maxFingerprintSize, a2maxFingerprintSize2), 17);
      //cout << "for f1, a1 bound: " << i << ", " << alpha1 << " the eventual bound is " << a2maxFingerprintSize << endl;
      if (fprForFingerPrintSizeCF(a2maxFingerprintSize) > 0.5) {
        continue;
      }
      for (int j = c+1; j <= a2maxFingerprintSize; j++) {
        OptimizationObject newOptimizationObject = createNewOptimizationObject(i, j, currentOptObject);
        double variation = getTotalLogVariation(i,j,currentOptObject, true);
        incrementBuckets(buckets, variation);
        //cout << "f1: " << i << ", f2: " << j << endl;
        myqueue.push(newOptimizationObject);
      }
    }

    myqueue.pop();
  }

  bucketPrint(buckets);

  cout << "total number of setups searched " << totalSearched << endl;
  cout << "FPR of best setup: " << bestFPR << endl;
  return bestSetup.fingerprints;

}

int main(int argc, char *argv[]) {
  // argument 1: minimization of fpr or space
  // argument 2: constraint (either FPR or space in bits per element)
  // argument 3: psi
  // argument 4: N / P
  // Example: ./filter_optimizer fpr 10 0.8 2
  // translation: minimize fpr for a stacked filter using less than 10 bits per element, given psi=0.8, N/P = 2

  if (argc < 6) {
    cout << "There should be the following 4 arguments: the minimization (fpr or space),"
    "the constraint (the other of fpr or space), psi, N/P, multiplicative slack " << endl;
    cout << "Ex: ./filter_optimizer fpr 10 0.8 2 0.001" << endl;
    return 0;
  }

  std::string minimization = argv[1];
  double constraint = std::stod(argv[2]);
  double psi = std::stod(argv[3]);
  double NoverP = std::stod(argv[4]);
  double multiplicativeSlack = std::stod(argv[5]);
  double logGoal = log2(multiplicativeSlack);
  cout << printf("%s, %f, %f, %f %f", minimization.c_str(), constraint, psi, NoverP, multiplicativeSlack) << endl;
  cout << printf("logGoal: %f", logGoal) << endl;

  // ignore size minimization for given FPR for now. Focus on size -> minimum FPR

  // 1) calculate single layer FPR. Use this to get epsilon slack for best FPR allowed. 
  int maxFingerprintSize = floor(constraint * 0.95);
  cout << (constraint * 0.95) << ", " << maxFingerprintSize << endl;
  double bestFPR = fprForFingerPrintSizeCF(maxFingerprintSize);
  double epsilonSlack = bestFPR * multiplicativeSlack;
  cout << "slack is: " << epsilonSlack << endl;

  /*OptimizationObject startObject(constraint,psi, NoverP, epsilonSlack);
  OptimizationObject bestbyFPR = optimizeFPRunderSizeConstraintDepthFirst(startObject);

  cout << "Best by Depth First " << endl;
  for (auto i: bestbyFPR.fingerprints) {
    std::cout << i << ' ';
  }
  cout << endl;
  cout << "FPR: " << bestbyFPR.FPRcurr << endl; */

  std::vector<int> returnSetup = optimizeFPRunderSizeConstraint(constraint, psi, NoverP, epsilonSlack);

  for (auto i: returnSetup)
    std::cout << i << ' ';
  cout << endl;

  return 0;
}

