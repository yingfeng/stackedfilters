#include <stdlib.h>
#include <fstream>
#include <random>
#include "../Headers/BloomFilter.h"
#include "../Headers/StackedAMQ.h"

std::vector<IntElement> generate_ints(int num_elements) {
  std::random_device rd;
  /* Random number generator */
  std::default_random_engine generator(rd());
  /* Distribution on which to apply the generator */
  std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
  std::vector<IntElement> el_vec(num_elements);
  for (int i = 0; i < num_elements; i++) {
    el_vec[i] = i;
  }
  std::random_shuffle(el_vec.begin(), el_vec.end());
  return el_vec;
}

int main(int arg_num, char** args) {
  std::ofstream fs;
  fs.open("Data/Stacked_Filter_Data.csv");
  fs << "Beta,Psi,Bits Available,Equal Fprs,Num Layers,Used Bits,Total "
        "FPR,Known FPR,UnknownFPR,Filter Checks For Positive,Filter Checks For "
        "Negative\n";
  int total_elements = 1000000;
  for (double beta = .15; beta < .21; beta += .35) {
    for (double psi = .4; psi < .71; psi += .35) {
      for (double bits = 10; bits < 11; bits += 2) {
        int num_positives = total_elements * beta;
        int num_known_negatives = total_elements * (1 - beta);
        int num_unknown_negatives = num_known_negatives;
        int total_size = num_positives * bits;
        for (int inner = 0; inner <= 1; inner++) {
          int num_layers = 3;
          bool flip_zeros = false;
          if (inner == 1) {
            flip_zeros = true;
          }
          printf("beta =%f, psi=%f, bits = %f, equal_fprs=%d, num_layers=%d\n",
                 beta, psi, bits, false, num_layers);
          double known_fpr = 0;
          double unknown_fpr = 0;
          double total_fpr = 0;
          double used_bits = 0;
          double checks_per_pos = 0;
          double checks_per_neg = 0;
          int num_reps = 3;
          for (int reps = 0; reps < num_reps; reps++) {
            std::vector<IntElement> ints = generate_ints(
                num_positives + num_known_negatives + num_unknown_negatives);
            std::vector<IntElement> positives = std::vector<IntElement>(
                ints.begin(), ints.begin() + num_positives);
            std::vector<IntElement> known_negatives = std::vector<IntElement>(
                ints.begin() + num_positives,
                ints.begin() + num_positives + num_known_negatives);
            std::vector<IntElement> unknown_negatives = std::vector<IntElement>(
                ints.begin() + num_positives + num_known_negatives, ints.end());
            StackedAMQ<BloomFilter, IntElement> filter(
                num_layers, positives, known_negatives, total_size, psi,
                .0000001, false);
            uint64 bits_flipped = 0;
            BloomFilter<IntElement>* n_filter = &filter.layer_array_[1];
            uint64 num_negatives_expected = 0;
            for (auto negative : known_negatives) {
              if (filter.layer_array_[0].LookupElement(negative))
                num_negatives_expected++;
            }
            n_filter->num_hashes_ =
                std::max(
                    (int)(round(-log(filter.layer_fprs_[1]) / log(2)) + .5),
                    1) *
                .34;
            n_filter->filter_size_ =
                1 /
                (1 - (double)pow(
                         (1 - (double)pow(filter.layer_fprs_[2],
                                          (double)1 / n_filter->num_hashes_)),
                         (double)1 /
                             (num_negatives_expected * n_filter->num_hashes_)));
            n_filter->filter_ = std::vector<bool>(n_filter->GetSize(), false);
            for (auto negative : known_negatives) {
              if (filter.layer_array_[0].LookupElement(negative))
                n_filter->InsertElement(negative);
            }
            std::vector<int> check_array(n_filter->GetSize(), 0);
            for (const auto positive : positives) {
              uint64 hash_1 = n_filter->getHash1(positive);
              uint64 hash_2 = n_filter->getHash2(positive);
              for (int k = 0; k < n_filter->num_hashes_; k++) {
                check_array[n_filter->getNthHash(hash_1, hash_2, k)]++;
              }
            }
            std::vector<bool> new_filter(n_filter->GetSize(), true);
            for (auto positive : positives) {
              uint64 hash_1 = n_filter->getHash1(positive);
              uint64 hash_2 = n_filter->getHash2(positive);
              int max_int = -1;
              uint64 max_loc = 0;
              for (int k = 0; k < n_filter->num_hashes_; k++) {
                uint64 hash_k = n_filter->getNthHash(hash_1, hash_2, k);
                if (new_filter[hash_k] == false) {
                  max_int = -1;
                  break;
                }
                if (check_array[hash_k] > max_int &&
                    n_filter->filter_[hash_k] == false) {
                  max_int = check_array[hash_k];
                  max_loc = hash_k;
                }
              }
              if (max_int < 0) {
                continue;
              }
              new_filter[max_loc] = false;
            }
            n_filter->filter_ = new_filter;
            uint64 num_expected_elements = 0;
            for (auto positive : positives) {
              if (n_filter->LookupElement(positive)) num_expected_elements++;
            }
            BloomFilter<IntElement>* p_filter = &filter.layer_array_[2];
            p_filter->num_hashes_ = std::max(
                (int)(round(-log(filter.layer_fprs_[2]) / log(2)) + .5), 1);
            p_filter->filter_size_ =
                1 /
                (1 - (double)pow(
                         (1 - (double)pow(filter.layer_fprs_[2],
                                          (double)1 / p_filter->num_hashes_)),
                         (double)1 /
                             (num_expected_elements * p_filter->num_hashes_)));
            p_filter->filter_ = std::vector<bool>(p_filter->GetSize(), 0);
            for (int i = 0; i < p_filter->GetSize(); i++) {
              p_filter->filter_[i] = 0;
            }
            for (const auto positive : positives) {
              if (n_filter->LookupElement(positive)) {
                p_filter->InsertElement(positive);
              }
            }

            if (!flip_zeros) {
              filter = StackedAMQ<BloomFilter, IntElement>(
                  num_layers, positives, known_negatives, filter.GetSize(), psi,
                  .0000001, false);
            }
            filter.PrintLayerDiagnostics();
            filter.ResetNumFilterChecks();
            for (int i = 0; i < num_positives; i++)
              if (!filter.LookupElement(positives[i])) printf("ERROR!!!");
            checks_per_pos += (double)filter.NumFilterChecks() / num_positives;
            filter.ResetNumFilterChecks();
            int kfp = 0;
            for (int i = 0; i < num_known_negatives; i++) {
              if (filter.LookupElement(known_negatives[i])) {
                kfp++;
              }
            }
            checks_per_neg +=
                (double)filter.NumFilterChecks() / num_known_negatives * psi;
            filter.ResetNumFilterChecks();
            int ukfp = 0;
            for (int i = 0; i < num_unknown_negatives; i++) {
              if (filter.LookupElement(unknown_negatives[i])) {
                ukfp++;
              }
            }
            checks_per_neg += (double)filter.NumFilterChecks() /
                              num_unknown_negatives * (1 - psi);
            known_fpr += (double)(kfp) / (double)(num_known_negatives);
            unknown_fpr += (double)(ukfp) / (double)(num_unknown_negatives);
            total_fpr +=
                psi * (double)(kfp) / (double)(num_known_negatives) +
                (1 - psi) * (double)(ukfp) / (double)(num_unknown_negatives);
            used_bits += (double)filter.GetSize() / num_positives;
            printf("Trial FPR:%f Bits_Flipped:%ld\n",
                   psi * (double)(kfp) / (double)(num_known_negatives) +
                       (1 - psi) * (double)(ukfp) /
                           (double)(num_unknown_negatives),
                   bits_flipped);
          }
          known_fpr = known_fpr / num_reps;
          unknown_fpr = unknown_fpr / num_reps;
          total_fpr = total_fpr / num_reps;
          used_bits = used_bits / num_reps;
          checks_per_neg = checks_per_neg / num_reps;
          checks_per_pos = checks_per_pos / num_reps;
          fs << beta << "," << psi << "," << bits << "," << false << ","
             << num_layers << "," << used_bits << "," << total_fpr << ","
             << known_fpr << "," << unknown_fpr << "," << checks_per_pos << ","
             << checks_per_neg << "\n";
          printf(", fpr=%f, unknown_fpr=%f,pos_checks=%f, neg_checks=%f",
                 total_fpr, unknown_fpr, checks_per_pos, checks_per_neg);
          printf(" Used Bits= %f\n", used_bits);
        }
      }
    }
  }
  fs.close();
}