#include <stdlib.h>
#include <fstream>
#include <random>
#include "../Headers/BloomFilter.h"
#include "../Headers/CityHash.h"

// Some thoughts... simple amplification doesn't seem to give a lot of boost in
// performance. It's not nothing, about a 10-15% savings in FPR, but not exactly
// groundbreaking. Now, if there were a family in which previously successful
// members of the family could point you towards more successful ones, then this
// might be more tractable. Perhaps look into locality perserving hash functions
// or just the broader family of hash functions? Further, the stacked filter
// design might see a more fruitful application of this because it likely has a
// higher variance in filter performance in the first place. Could a "worse"
// hash function benefit enough from this method to make up for the clustering
// of the negative hashes that would be induced as well?

std::vector<long> generate_ints2(int num_elements) {
  std::random_device rd;
  /* Random number generator */
  std::default_random_engine generator(rd());
  /* Distribution on which to apply the generator */
  std::uniform_int_distribution<long> distribution(0, 0xFFFFFFFFFFFFFFF);
  std::vector<long> int_vec(num_elements);
  for (int i = 0; i < num_elements; i++) {
    int_vec[i] = i;
  }
  std::random_shuffle(int_vec.begin(), int_vec.end());
  return int_vec;
}

size_t getHash1(long x, int seed) {
  return CityHash64WithSeed((char*)&x, 4, 1234567 + seed);
}

size_t getHash2(long x, int seed) {
  return CityHash64WithSeed((char*)&x, 4, 7654321 + seed);
}

// Double hashing strategy recommended in Mitzenmacher Paper.
size_t getNthHash(size_t hash1, size_t hash2, size_t hash_num,
                  size_t filter_size) {
  return (hash1 + hash_num * hash2) % filter_size;
}

int main(int arg_num, char** args) {
  int num_positives = 100000;
  int num_negatives = 100000;
  double simple_fpr = 0;
  double amp_fpr = 0;
  double simple_LF = 0;
  double amp_LF = 0;
  double prob_full_solo = 0;
  int num_reps = 10;
  double bits_per = 6;
  double bits_per_backup = 7;
  double bits_per_divider = 4;
  int num_hashes_original = std::max((int)(round(bits_per * log(2))), 1) + 5;
  int non_collisions = (int)((double)bits_per * log(2)) - 1;
  for (int outer = 0; outer < 1; outer++) {
    double standard_fpr_total = 0;
    double new_fpr_total = 0;
    for (int reps = 0; reps < num_reps; reps++) {
      int seed = rand();
      std::vector<long> ints = generate_ints2(num_positives + num_negatives);
      std::vector<unsigned int> filter_1(num_positives * bits_per);
      for (int i = 0; i < num_positives; i++) {
        size_t hash1 = getHash1(ints[i], seed);
        size_t hash2 = getHash2(ints[i], seed);
        for (int j = 0; j < num_hashes_original; j++)
          filter_1[getNthHash(hash1, hash2, j, filter_1.size())]++;
      }
      std::vector<unsigned int> num_solo(num_positives, 0);
      for (int i = 0; i < num_positives; i++) {
        size_t hash1 = getHash1(ints[i], seed);
        size_t hash2 = getHash2(ints[i], seed);
        for (int j = 0; j < num_hashes_original; j++)
          num_solo[i] +=
              (filter_1[getNthHash(hash1, hash2, j, filter_1.size())] == 1);
      }

      int full_solo = 0;
      for (int i = 0; i < num_positives; i++)
        if (num_solo[i] >= non_collisions) full_solo++;
      printf("Full Solo Prob:%f ", (double)full_solo / num_positives);
      std::vector<bool> filter_original(num_positives * bits_per, 0);
      std::vector<bool> filter_divider(full_solo * bits_per_divider, 0);
      int num_hashes_backup =
          std::max((int)(round(bits_per_backup * log(2))), 1);
      int num_hashes_divider =
          std::max((int)(round(bits_per_divider * log(2))), 1);
      size_t num_elements_in_backup = 0;
      for (int i = 0; i < num_positives; i++) {
        if (num_solo[i] >= non_collisions) {
          size_t hash1 = getHash1(ints[i], seed);
          size_t hash2 = getHash2(ints[i], seed);
          for (int j = 0; j < num_hashes_divider; j++)
            filter_divider[getNthHash(hash1, hash2, j, filter_divider.size())] =
                1;
        }
      }

      for (int i = 0; i < num_positives; i++) {
        size_t hash1 = getHash1(ints[i], seed);
        size_t hash2 = getHash2(ints[i], seed);
        bool in_backup_filter = true;
        for (int j = 0; j < num_hashes_divider; j++) {
          if (!filter_divider[getNthHash(hash1, hash2, j,
                                         filter_divider.size())]) {
            in_backup_filter = false;
          }
        }
        num_elements_in_backup += in_backup_filter;
      }
      std::vector<bool> filter_backup(num_elements_in_backup * bits_per_backup,
                                      0);
      std::vector<bool> filter_standard(
          num_positives * bits_per + bits_per_divider * full_solo +
              bits_per_backup * num_elements_in_backup,
          0);
      int num_hashes_standard = std::max(
          (int)(round((double)(num_positives * bits_per +
                               bits_per_divider * full_solo +
                               bits_per_backup * num_elements_in_backup) /
                      num_positives * log(2))),
          1);
      ;
      for (int i = 0; i < num_positives; i++) {
        size_t hash1 = getHash1(ints[i], seed);
        size_t hash2 = getHash2(ints[i], seed);
        for (int j = 0; j < num_hashes_standard; j++)
          filter_standard[getNthHash(hash1, hash2, j, filter_standard.size())] =
              1;
        bool in_backup_filter = true;
        for (int j = 0; j < num_hashes_divider; j++) {
          if (!filter_divider[getNthHash(hash1, hash2, j,
                                         filter_divider.size())]) {
            in_backup_filter = false;
          }
        }
        if (in_backup_filter) {
          for (int j = 0; j < num_hashes_backup; j++)
            filter_backup[getNthHash(hash1, hash2, j, filter_backup.size())] =
                1;
        } else {
          for (int j = 0; j < num_hashes_original; j++)
            filter_original[getNthHash(hash1, hash2, j,
                                       filter_original.size())] = 1;
        }
      }
      double standard_fpr = 0;
      double original_fpr = 0;
      double divider_fpr = 0;
      double backup_fpr = 0;
      double new_fpr = 0;
      for (int i = 0; i < num_negatives; i++) {
        size_t hash1 = getHash1(ints[num_positives + i], seed);
        size_t hash2 = getHash2(ints[num_positives + i], seed);
        bool standard_fps = true;
        bool in_backup_filter = true;
        bool original_fps = true;
        bool divider_fps = true;
        bool backup_fps = true;
        for (int j = 0; j < num_hashes_standard; j++) {
          if (filter_standard[getNthHash(hash1, hash2, j,
                                         filter_standard.size())] == 0) {
            standard_fps = false;
            break;
          }
        }
        for (int j = 0; j < num_hashes_divider; j++) {
          if (!filter_divider[getNthHash(hash1, hash2, j,
                                         filter_divider.size())]) {
            in_backup_filter = false;
            divider_fps = false;
            break;
          }
        }
        for (int j = 0; j < num_hashes_original; j++) {
          if (filter_original[getNthHash(hash1, hash2, j,
                                         filter_original.size())] == 0) {
            original_fps = false;
            break;
          }
        }
        for (int j = 0; j < num_hashes_backup; j++) {
          if (filter_backup[getNthHash(hash1, hash2, j,
                                       filter_backup.size())] == 0) {
            backup_fps = false;
            break;
          }
        }
        if (backup_fps == true) backup_fpr++;
        if (original_fps == true) original_fpr++;
        if (divider_fps == true) divider_fpr++;
        if ((in_backup_filter && backup_fps == true) ||
            (!in_backup_filter && original_fps == true))
          new_fpr++;
        if (standard_fps == true) standard_fpr++;
      }
      standard_fpr /= num_negatives;
      original_fpr /= num_negatives;
      divider_fpr /= num_negatives;
      backup_fpr /= num_negatives;
      new_fpr /= num_negatives;
      double standard_LF = 0;
      double original_LF = 0;
      double divider_LF = 0;
      double backup_LF = 0;
      for (int i = 0; i < filter_standard.size(); i++)
        standard_LF += filter_standard[i];
      for (int i = 0; i < filter_original.size(); i++)
        original_LF += filter_original[i];
      for (int i = 0; i < filter_divider.size(); i++)
        divider_LF += filter_divider[i];
      for (int i = 0; i < filter_backup.size(); i++)
        backup_LF += filter_backup[i];
      standard_LF /= filter_standard.size();
      original_LF /= filter_original.size();
      divider_LF /= filter_divider.size();
      backup_LF /= filter_backup.size();
      printf(
          "TRIAL %d standard_fpr:%f original_fpr:%f divider_fpr:%f "
          "backup_fpr:%f\n "
          "standard_LF:%f "
          "original_LF:%f divider_LF%f backup_LF:%f\n",
          reps, standard_fpr, original_fpr, divider_fpr, backup_fpr,
          standard_LF, original_LF, divider_LF, backup_LF);
      printf("original_size:%ld divider_size:%ld backup_size:%ld\n",
             filter_original.size(), filter_divider.size(),
             filter_backup.size());
      standard_fpr_total += standard_fpr;
      new_fpr_total += new_fpr;
    }
    standard_fpr_total /= num_reps;
    new_fpr_total /= num_reps;
    printf("standard_fpr:%f new_fpr:%f\n", standard_fpr_total, new_fpr_total);
  }
}