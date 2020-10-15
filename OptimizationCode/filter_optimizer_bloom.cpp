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

int TEN_MILLION = 10000000;
int ONE_MILLION = 1000000;
int ONE_THOUSAND = 1000;
int TEN_THOUSAND = 10000;
int ONE_HUNDRED_THOUSAND = 100000;

class OptimizationObject {
  public:
    double constraint;
    double psi;
    double NoverP;
    double slack;
    double FPRcurr;
    double proportion;
    int countBelow;
    std::vector<double> bitsPerElementLayers;
    OptimizationObject();
    OptimizationObject(double constraint, double psi, double NoverP, double slack, double FPRcurr, double proportion, std::vector<double> fingerprints);
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
      double FPRcurr, double proportion, std::vector<double> bitsPerElementLayers) {
  this->constraint = constraint;
  this->psi = psi;
  this->NoverP = NoverP;
  this->slack = slack;
  this->FPRcurr = FPRcurr;
  this->proportion = proportion;
  this->bitsPerElementLayers = bitsPerElementLayers;
}

double removeLoadFactorSize(double constraintSize) {
  return pow(log(2),2) * constraintSize;
}

// The following two methods do not include the load factor (1 / (ln 2)^2)
double fprForBitsPerEleBFNoLoad(double bitsPerEle) {
  return exp(-1 * bitsPerEle);
}

double sizeForFprBFNoLoad(double inputFPR) {
  return -log(inputFPR);
}

// the following three methods include the load factor in calculations. 

double fprForBitsPerEleBF(double bitsPerEle) {
  return min(1.0, pow(2, -1 * log(2) * bitsPerEle));
}

double sizeForFprBF(double inputFPR) {
  return -log(inputFPR) * (1.0 / pow(log(2), 2));
}

OptimizationObject createNewOptimizationObject(double bitsEle1, double bitsEle2, OptimizationObject currentOptObject) {
  OptimizationObject returnObject;
  double alpha1 = fprForBitsPerEleBF(bitsEle1);
  double alpha2 = fprForBitsPerEleBF(bitsEle2);
  returnObject.constraint = (currentOptObject.constraint - bitsEle1 - currentOptObject.NoverP * alpha1 * bitsEle2) / alpha2;
  returnObject.psi = currentOptObject.psi / (currentOptObject.psi + ((1 - currentOptObject.psi) * alpha2));
  returnObject.NoverP = currentOptObject.NoverP * (alpha1 / alpha2);
  returnObject.slack = currentOptObject.slack;
  returnObject.proportion = currentOptObject.proportion * alpha1 * (currentOptObject.psi + (1 - currentOptObject.psi) * alpha2);
  returnObject.FPRcurr = currentOptObject.FPRcurr + (1 - currentOptObject.psi) * alpha1 * (1 - alpha2);
  returnObject.bitsPerElementLayers = currentOptObject.bitsPerElementLayers;
  returnObject.bitsPerElementLayers.push_back(bitsEle1);
  returnObject.bitsPerElementLayers.push_back(bitsEle2);
  return returnObject;
}

void printBitsEles(OptimizationObject currentOptObject) {
  cout << "inputFingerprints ";
  for (auto i: currentOptObject.bitsPerElementLayers)
    std::cout << i << ' ';
  std::cout << endl;
}

double calculateSize(std::vector<double> bitsPerElementLayers, double NoverP) {
  double posSize = 0.0;
  double negSize = 0.0;
  for (unsigned int i = 0; i < bitsPerElementLayers.size(); i++) {
    if (i % 2 == 0) {
      double proportion = 1.0;
      for (unsigned int j = 0; j < i/2; j++) {
        proportion = proportion * fprForBitsPerEleBF(bitsPerElementLayers[(2 * j) + 1]);
      }
      posSize += proportion * bitsPerElementLayers[i];
    } else {
      double proportion = 1.0;
      for (unsigned int j = 0; j < (i+1)/2; j++) {
        proportion = proportion * fprForBitsPerEleBF(bitsPerElementLayers[2 * j]);
      }
      negSize += NoverP * proportion * bitsPerElementLayers[i];
    }
  }
  double totalSize = posSize + negSize;
  return totalSize;
}

double equal_fpr_gradientForSize(double NoverP, double alpha) {
  double numeratorPart1 = (alpha - 1) * ((NoverP * alpha) + 1);
  double numeratorPart2 = -1 * (NoverP + 1) * alpha * log(alpha);
  double denominator = pow(1 - alpha, 2) * alpha;

  double unscaledGradient = (numeratorPart1 + numeratorPart2) / denominator;
  double scaledGradient = (1 / pow(log(2), 2)) * unscaledGradient;
  return scaledGradient;
}

double sizeForFixedAlpha(double NoverP, double alpha) {
  return (-1 / pow(log(2), 2)) * log(alpha) * ((1 + NoverP * alpha) / (1 - alpha));
}

double fprFixedAlpha(double psi, double alpha) {
  //return (1 - psi) * (alpha / (alpha + 1));
  return (1 - psi) * (1 -alpha) * (alpha / 1 - pow(alpha, 2));
}

double optimizeFPRunderSizeConstraintLayerEqual(double constraint, double psi, double NoverP, double initialAlpha, double epsilonGradient) {
  double stepSize = 1;
  double alpha = initialAlpha;
  double size = sizeForFixedAlpha(NoverP, alpha);
  int count = 0;
  while (stepSize > epsilonGradient) {
    double gradient = equal_fpr_gradientForSize(NoverP, alpha);
    if (size > constraint) {
      double newAlpha = alpha - (stepSize * gradient);
      double newSize = sizeForFixedAlpha(NoverP, newAlpha);
      while (newSize > size or newAlpha < 0 or newAlpha > 1) {
        stepSize = stepSize * 0.5;
        newAlpha = alpha - (stepSize * gradient);
        if (stepSize < epsilonGradient) {
          break;
        }
        newSize = sizeForFixedAlpha(NoverP, newAlpha);
      }
      alpha = newAlpha;
      size = sizeForFixedAlpha(NoverP, newAlpha);
    } else {
      double newAlpha = alpha - stepSize;
      while (sizeForFixedAlpha(NoverP, newAlpha) > constraint or newAlpha < 0) {
        stepSize = stepSize * 0.5;
        newAlpha = alpha - stepSize;
        if (stepSize < epsilonGradient) {
          break;
        }
      }
      size = sizeForFixedAlpha(NoverP, newAlpha);
      alpha = newAlpha;
    }
    count += 1;
  }
  cout << "size of optimalBF: " << sizeForFixedAlpha(NoverP, alpha) << endl;
  cout << "FPR of optimalBF: " << fprFixedAlpha(psi, alpha) << endl;
  cout << "# of gradient iterations: " << count << endl;
  if (sizeForFixedAlpha(NoverP, alpha) > constraint + 0.01) {
    cout << "alpha: " << alpha << endl;
    alpha = -1.0;
  }
  cout << "alpha value is: " << alpha << endl;
  return alpha;
}
 
std::vector<double> optimizeFPRunderSizeConstraintDiscreteSearch(double sizeBudget, double psi, double NoverP, double epsilonSlack, double bitStep) {

  std::queue<OptimizationObject> myqueue;
  OptimizationObject startObject(sizeBudget,psi, NoverP, epsilonSlack);
  OptimizationObject bestSetup = startObject;
  // myqueue.push(OptimizationObject());
  myqueue.push(startObject);
  long long int totalSearched = 0;
  // should be overwritten immediately. 
  double bestFPR = 1.1;
  while(!myqueue.empty()) {
    OptimizationObject& currentOptObject = myqueue.front();
    totalSearched += 1;
    if (totalSearched % TEN_MILLION == 0) {
      cout << "ten millions: " << totalSearched / TEN_MILLION << endl << std::flush;
    }
    if (currentOptObject.FPRcurr > bestFPR) {
      myqueue.pop();
      continue;
    } 

    // choose 1 layer
    double a1sizeMax = currentOptObject.constraint;
    double addedFPR = currentOptObject.proportion * fprForBitsPerEleBF(a1sizeMax);
    if (currentOptObject.FPRcurr + addedFPR < bestFPR) {
      bestFPR = currentOptObject.FPRcurr + addedFPR;
      bestSetup = currentOptObject;
      bestSetup.bitsPerElementLayers.push_back(a1sizeMax);
    }

    // check if a single layer solution puts us within epsilon of best possible using the current stack
    // if all queries on negatives were caught, difference would be less than epsilon. Stop here.    
    if (addedFPR < epsilonSlack) {
      myqueue.pop();
      continue;
    }
    
    // loop through multi-layer solutions
    // calculate alpha_max for layer 1
    double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * log(2))), fprForBitsPerEleBF(a1sizeMax) / (0.5 * (1 - currentOptObject.psi)));
    double a1sizeMin = sizeForFprBF(alpha_max);
    if (a1sizeMin <= 0) {
      a1sizeMin = sizeForFprBF(0.5);
    }
    if (totalSearched < 0) {
      cout << "a1sizeMax - a1sizeMin / increment: " << (a1sizeMax - a1sizeMin) / 0.1 << endl;
    }
    for (double i = a1sizeMin; i <= a1sizeMax; i+= bitStep) {
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
      for (double j = sizeForFprBF(0.5); j <= a2maxBitsPerEle; j+= bitStep) {
        OptimizationObject newOptimizationObject = createNewOptimizationObject(i, j, currentOptObject);
        myqueue.push(newOptimizationObject);
      }
    }

    myqueue.pop();
  }

  cout << "total number of setups searched " << totalSearched << endl;
  cout << "FPR of best setup: " << bestFPR << endl;
  return bestSetup.bitsPerElementLayers;

}

/*void optimizeFPRunderSizeConstraintSeed(double sizeBudget, double psi, double NoverP, double epsilonSlack, double spacing, &int totalSearched, &std::vector<OptimizationObject> searchList) {
  std::queue<OptimizationObject> myqueue;
  OptimizationObject startObject(sizeBudget,psi, NoverP, epsilonSlack);
  OptimizationObject bestSetup = startObject;
  myqueue.push(startObject);
  // should be overwritten immediately. 
  double bestFPR = 1.1;

  while(!myqueue.empty()) {
    OptimizationObject& currentOptObject = myqueue.front();
    totalSearched += 1;
    if (totalSearched % ONE_MILLION == 0) {
      cout << "millions: " << totalSearched / ONE_MILLION << endl << std::flush;
    }
    if (calculateCurrentFPR(psi, currentOptObject.bitsPerElementLayers, spacing) > bestFPR + epsilonSlack) {
      myqueue.pop();
      continue;
    } 

    // choose 1 layer
    double a1sizeMax = currentOptObject.constraint;
    double addedFPR = currentOptObject.proportion * fprForBitsPerEleBF(a1sizeMax);
    if (currentOptObject.FPRcurr + addedFPR < bestFPR) {
      bestFPR = currentOptObject.FPRcurr + addedFPR;
    }
    // check if a single layer solution puts us within epsilon of best possible using the current stack
    // if all queries on negatives were caught, difference would be less than epsilon. Stop here. 
    
    if (addedFPR < epsilonSlack) {
      if (calculateCurrentFPR(psi, currentOptObject.bitsPerElementLayers, spacing) < bestFPR + epsilonSlack) {
        searchList.push_back(currentOptObject);
      }
      myqueue.pop();
      continue;
    }
    
    // loop through multi-layer solutions
    // calculate alpha_max for layer 1
    double alpha_max = min(min(0.5, 1.0 / (currentOptObject.NoverP * log(2))), fprForBitsPerEleBF(a1sizeMax) / (0.5 * (1 - currentOptObject.psi)));
    double a1sizeMin = sizeForFprBF(alpha_max);
    if (a1sizeMin <= 0) {
      a1sizeMin = sizeForFprBF(0.5);
    }
    for (double i = a1sizeMin; i <= a1sizeMax; i+= 0.25) {
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
      for (double j = sizeForFprBF(0.5); j <= a2maxBitsPerEle; j+=0.25) {
        OptimizationObject newOptimizationObject = createNewOptimizationObject(i, j, currentOptObject);
        myqueue.push(newOptimizationObject);
      }
    }

    myqueue.pop();
  }

} */

int main(int argc, char *argv[]) {
  // argument 1: minimization of fpr or space
  // argument 2: constraint (either FPR or space in bits per element)
  // argument 3: psi
  // argument 4: N / P
  // Example: ./filter_optimizer fpr 10 0.8 2 0.001
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
  double bestFPR = fprForBitsPerEleBF(constraint);
  cout << "one layer FPR is: " << bestFPR << endl;
  double epsilonSlack = bestFPR * multiplicativeSlack;
  cout << "slack is: " << epsilonSlack << endl;

  auto startEqual = high_resolution_clock::now(); 
  optimizeFPRunderSizeConstraintLayerEqual(constraint, psi, NoverP, 0.001, 0.000001);
  auto stopEqual = high_resolution_clock::now();
  auto durationEqual = duration_cast<microseconds>(stopEqual - startEqual);
  cout << "duration equal is: " << durationEqual.count() << " microseconds" << endl;

  auto start = high_resolution_clock::now(); 
  std::vector<double> returnSetup = optimizeFPRunderSizeConstraintDiscreteSearch(constraint, psi, NoverP, epsilonSlack, 0.1);
  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "duration is: " << duration.count() << " milliseconds" << endl;

  for (auto i: returnSetup)
    std::cout << i << ' ';
  cout << endl;

  cout << "size of best setup: " << calculateSize(returnSetup, NoverP) << endl;

  return 0;
}

