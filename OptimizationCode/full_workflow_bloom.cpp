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
#include "nlopt.h"

#include "workload_generator.h"
#include "OptimizationRoutines.h"

int ONE_MILLION = 1000000;
int ONE_THOUSAND = 1000;
int TEN_THOUSAND = 10000;
int ONE_HUNDRED_THOUSAND = 100000;

/*double fprForBitsPerEleBF(double bitsPerEle) {
  return min(1.0, pow(2, -1 * log(2) * bitsPerEle));
}

double sizeForFprBF(double inputFPR) {
  return -log(inputFPR) * (1.0 / pow(log(2), 2));
} */

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

double calculateEFPR(std::vector<double> bitsPerElementLayers, double psi) {
  double proportion = 1.0;
  double FPRcurr = 0.0;
  double psiTrack = psi;
  for (unsigned int i = 0; i < bitsPerElementLayers.size()/2; i++) {
    double alpha1 = min(1.0, pow(2, -1 * log(2) * bitsPerElementLayers[i]));
    double alpha2 = min(1.0, pow(2, -1 * log(2) * bitsPerElementLayers[i+1]));
    proportion = proportion * alpha1 * (psi + (1 - psi) * alpha2);
    FPRcurr = FPRcurr + (1 - psiTrack) * alpha1 * (1 - alpha2);
    psiTrack = psiTrack / (psiTrack + ((1 - psiTrack) * alpha2));
  }
  if (bitsPerElementLayers.size() % 2 > 0) {
    FPRcurr += proportion * min(1.0, pow(2, -1 * log(2) * bitsPerElementLayers[bitsPerElementLayers.size()-1]));
  } else {
    FPRcurr += proportion;
  }
  return FPRcurr;
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

// derivation: at infinity layers, known negatives are all caught. For unknown negatives is (1 - alpha) * sum_i=0^inf alpha^(2i+1)
// this is then (1 - alpha) * alpha * sum_i=0^inf (x^2)^i -> traditional power series. 
// these two equations are the same. 
double fprFixedAlpha(double psi, double alpha) {
  return (1 - psi) * (alpha / (alpha + 1));
  //return (1 - psi) * (1 -alpha) * (alpha / 1 - pow(alpha, 2));
}

double optimizeFPRunderSizeConstraintLayerEqual(double constraint, double NoverP, double initialAlpha, double epsilonGradient) {
  double stepSize = 1;
  double alpha = initialAlpha;
  double size = sizeForFixedAlpha(NoverP, alpha);
  int count = 0;
  while (stepSize > epsilonGradient) {
    double gradient = equal_fpr_gradientForSize(NoverP, alpha);
    if (size > constraint) {
      if (gradient * stepSize == 0) {
        break;
      }
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
  if (sizeForFixedAlpha(NoverP, alpha) > constraint + 0.01) {
    alpha = -1.0;
  }
  return alpha;
}

double calculateEFPRFinite(double alpha, double psi, int numLayers) {
  double EFPRNf = psi * pow(alpha, (numLayers + 1) / 2);
  double EFPRNi = (1 - psi) * (alpha + pow(alpha, numLayers + 1)) / (1 + alpha);
  return EFPRNf + EFPRNi;
}

int numLayersTruncation(double alpha, double psi, double EFPRInfinite, double epsilon) {
  int Tl = 1;
  while (calculateEFPRFinite(alpha, psi, Tl) > EFPRInfinite + epsilon) {
    Tl += 2;
    if (Tl > 30) {
      cout << "problem in truncation likely" << endl;
      break;
    }
  }
  return Tl;
}

std::vector<double> optimizeStackedFilterBloom(double maxSize, unsigned int numPositiveElements, double epsilonSlack, std::vector<double>& psiPerNegativeElement) {
  // 1) calculate single layer FPR
  double bestFPR = fprForBitsPerEleBF(maxSize);
  double originalFPR = bestFPR;
  cout << "base FPR is " << originalFPR << endl;
  cout << "slack is: " << epsilonSlack << endl;

  vector<double> bestSetup;

  double bestFPREqual = 1.1;
  double bestAlpha = 1.0;

  double lastPsi = 0;
  double psiUsed = 0.0;

  // sweep through N_samp
  for (unsigned int i = 1; i <= psiPerNegativeElement.size(); i+= 50) {
    double newPsi = psiPerNegativeElement[i-1];
    double slackFromN = (newPsi - lastPsi) * min(originalFPR / (1 - newPsi), 1.0);
    if (slackFromN > epsilonSlack / 3) {
      double NoverP = static_cast<double>(i) / static_cast<double>(numPositiveElements);
      double alphaBest = optimizeFPRunderSizeConstraintLayerEqual(maxSize, NoverP, 0.01, 0.0001);
      lastPsi = newPsi;
      if (alphaBest > 0 and fprFixedAlpha(newPsi, alphaBest) < bestFPREqual) {
        bestFPREqual = fprFixedAlpha(newPsi, alphaBest);
        bestAlpha = alphaBest;
        psiUsed = newPsi;
      }
    }
  }

  // truncate Stacked Filter (if infinite better than single layer)
  std::vector<double> returnSetup;
  if (bestFPREqual < fprForBitsPerEleBF(maxSize)) {
    unsigned int numLayers = numLayersTruncation(bestAlpha, psiUsed, bestFPREqual, epsilonSlack / 3);
    for (unsigned int i = 0; i < numLayers; i++) {
      returnSetup.push_back(bestAlpha);
    }
  } else {
    returnSetup.push_back(maxSize);
  }
  return returnSetup;
}

int main(int argc, char *argv[]) {
  if (argc < 8) {
    cout << "There should be the following 6 arguments: the minimization (fpr or space),"
    "the constraint (the other of fpr or space), multiplicative slack, positive elements (in millions), negative elements (in millions), max # negs to take, zipf" << endl;
    cout << "Ex: ./full_workflow_filter_optimizer fpr 10.1 0.001 1.2 10 2.0 1.0" << endl;
    return 0;
  }

  std::string minimization = argv[1];
  double constraint = std::stod(argv[2]);
  double epsilonSlack = std::stod(argv[3]);
  unsigned int numPositiveElements = static_cast<int>(std::stod(argv[4]) * ONE_MILLION);
  unsigned int numNegativeElements = static_cast<int>(std::stod(argv[5]) * ONE_MILLION);
  unsigned int maxNumNegativeElements = static_cast<int>(std::stod(argv[6]) * ONE_MILLION);
  assert(numNegativeElements > (numPositiveElements / 1000));
  double zipf = std::stod(argv[7]);

  ZipfianDistribution dist = ZipfianDistribution(zipf, numNegativeElements);
  std::vector<double> truncatedPsis;
  for (unsigned int i = 0; i < maxNumNegativeElements; i++) {
    truncatedPsis.push_back(dist.psiVals[i]);
  }
  std::vector<double> filter = optimizeStackedFilterBloom(constraint, numPositiveElements, epsilonSlack, truncatedPsis);
  for (unsigned int i = 0; i < filter.size(); i++) {
    cout << filter[i] << endl;
  }
  return 0;
}