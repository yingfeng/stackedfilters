#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-flp30-c"
//
// Created by kylebd99 on 8/30/19.
//

#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <unistd.h>
#include <random>
#include <string>
#include <chrono>
#include <tclap/CmdLine.h>
#include "BloomFilterLayer.h"
#include "StackedFilter.h"
#include "ZipfDistribution.h"

static const char alphanum[] =
        "0123456789"
        "!@#$%^&*"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

static const std::array<std::string, 6> url_suffix = {".com", ".net", ".org", ".ao", ".ph", ".ki"};

int alphaNumLength = sizeof(alphanum) - 1;

std::vector<StringElement> genRandomURLElementList(uint64 num_elements) {
    std::vector<StringElement> url_list;
    for (uint64 i = 0; i < num_elements; i++) {
        std::string rand_str = "www.";
        while (rand() % 100 < 90) {
            rand_str += alphanum[rand() % alphaNumLength];
        }
        StringElement url = rand_str + url_suffix[rand() % url_suffix.size()];
        url_list.push_back(url);
    }
    return url_list;
}

std::vector<StringElement> get_positive_urls() {
    std::vector<StringElement> blacklisted_urls(1713331);
    std::ifstream url_file("Data/positive_urls.csv");
    std::string line;
    int number_of_urls_gathered = 0;
    std::getline(url_file, line); // Skip the header
    while (std::getline(url_file, line) && number_of_urls_gathered < 1713331) {
        blacklisted_urls[number_of_urls_gathered] = line;
        number_of_urls_gathered++;
    }
    return blacklisted_urls;
}

std::vector<StringElement> get_negative_urls() {
    std::vector<StringElement> blacklisted_urls(10000000);
    std::ifstream url_file("Data/top10milliondomains_CLEANED.csv");
    std::string line;
    int number_of_urls_gathered = 0;
    std::getline(url_file, line); // Skip the header
    while (std::getline(url_file, line) && number_of_urls_gathered < 10000000) {
        int start = line.find(','); //Skip first number
        start = line.find(',', start + 1); //Skip the second number
        auto url = line.substr(start + 1, line.find(',', start + 1) - start - 1);
        blacklisted_urls[number_of_urls_gathered] = url;
        number_of_urls_gathered++;
    }
    blacklisted_urls.resize(number_of_urls_gathered);
    return blacklisted_urls;
}

std::pair<std::vector<double>, std::vector<double>> get_cdf_and_pmf() {
    std::vector<double> pmf(10000000);
    std::ifstream url_file("Data/top10milliondomains.csv");
    std::string line;
    int number_of_urls_gathered = 0;
    std::getline(url_file, line); // Skip the headers
    while (std::getline(url_file, line)) {
        int start = line.find(','); //Skip first number
        start = line.find(',', start + 1); //Skip the second number
        start = line.find(',', start + 1); //Skip the URL
        auto page_rank_str = line.substr(start + 1);
        pmf[number_of_urls_gathered] = pow(8, std::strtod(page_rank_str.c_str(), nullptr));
        number_of_urls_gathered++;
    }
    double page_rank_sum = 0;
    for (auto rank : pmf) {
        page_rank_sum += rank;
    }

    for (auto &rank : pmf) {
        rank = rank / page_rank_sum;
    }

    std::vector<double> cdf(pmf.size());
    cdf[0] = pmf[0];
    for (uint64 i = 1; i < pmf.size(); i++) {
        cdf[i] = cdf[i - 1] + pmf[i];
    }

    return {cdf, pmf};
}

void GenerateDataForOneRun(std::ofstream &file_stream, uint64 negative_universe_size,
                           const double bits_per_positive_element,
                           const uint32 num_reps, const uint64 max_known_negatives,
                           uint64 sample_size, const std::vector<StringElement> &positives,
                           const std::vector<StringElement> &negatives,
                           const std::vector<double> &cdf, const std::vector<double> &pmf) {
    double total_fpr = 0;
    double used_bits = 0;
    uint64 number_of_chosen_negatives = 0;
    long double psi = 0;
    uint64 num_positive_elements = positives.size();
    uint64 total_size = num_positive_elements * bits_per_positive_element;
    printf("num_positive_elements =%ld, negative_universe_size=%ld, bits = %f\n",
           positives.size(), negative_universe_size, bits_per_positive_element);
    for (uint32 reps = 0; reps < num_reps; reps++) {
        // Limit the negatives to the max number allowed
        std::vector<StringElement> potential_known_negatives = std::vector<StringElement>(
                negatives.begin(),
                negatives.begin() + max_known_negatives);
        std::vector<double> potential_known_negatives_cdf = std::vector<double>(
                cdf.begin(),
                cdf.begin() + max_known_negatives);
        StackedFilter<BloomFilterLayer, StringElement> filter(total_size, positives,
                                                              potential_known_negatives, potential_known_negatives_cdf);
        used_bits += filter.GetSize() / (double) num_positive_elements;
        filter.PrintLayerDiagnostics();
        uint64 num_known_negative_elements = filter.num_negative_;

        // Test the false positive rate for known negatives.
        uint64 false_positives = 0;
        uint64 known_negatives_tested = 0;

        // Generate the sample queries.
        std::minstd_rand gen(std::random_device{}());
        std::discrete_distribution<uint64_t> dist(pmf.begin(), pmf.end());
        std::vector<StringElement> elements_to_test;
        for (uint64 i = 0; i < sample_size; i++) {
            uint64_t element_rank = dist(gen);
            elements_to_test.push_back(negatives[element_rank]);
            if (element_rank < num_known_negative_elements) {
                known_negatives_tested++;
            }
        }

        // Filter Timing
        for (const auto &element : elements_to_test) {
            const bool is_false_positive = filter.LookupElement(element);
            if (is_false_positive) {
                false_positives++;
            }
        }

        total_fpr += (double) false_positives / (double) sample_size;
        psi += (double) known_negatives_tested / sample_size;
        printf("Trial FPR:%f, Negative Elements:%ld, approx psi:%f\n",
               (double) false_positives / (double) sample_size,
               num_known_negative_elements, (double) known_negatives_tested / sample_size);
    }
    total_fpr /= num_reps;
    used_bits /= num_reps;
    number_of_chosen_negatives /= num_reps;
    psi /= num_reps;


    file_stream << num_positive_elements << "," <<
                sample_size << "," << negative_universe_size << "," << max_known_negatives << ","
                << number_of_chosen_negatives << "," << psi << ","
                << bits_per_positive_element << "," << used_bits << "," << total_fpr << "\n";
    printf(", fpr=%f,Used Bits= %f\n",
           total_fpr, used_bits);
}

void set_flags(TCLAP::CmdLine &cmdLine, double &zipf_begin, double &zipf_max, double &zipf_step, double &bits_begin,
               double &bits_max, double &bits_step, uint64 &neg_universe_size,
               uint64 &neg_universe_size_max, double &neg_universe_size_ratio, uint64 &max_known_negatives,
               uint64 &max_known_negatives_max, double &max_known_negatives_ratio,
               uint32 &num_reps, std::string &file_path, int arg_num, char **args) {
    TCLAP::ValueArg<double> zipf_arg("", "zipf",
                                     "The starting value for the zipf parameter, and the only value if zipf_max is not set.",
                                     false, .75, "Expects a float.");
    TCLAP::ValueArg<double> zipf_max_arg("", "zipf_max",
                                         "The max value for the zipf parameter.", false, 10, "Expects a float.");
    TCLAP::ValueArg<double> zipf_step_arg("", "zipf_step",
                                          "The step size over which [zipf, zipf_max) will be explored.",
                                          false, 100, "Expects a float.");
    TCLAP::ValueArg<double> bits_begin_arg("", "bits",
                                           "The starting value for the bits per positive element, and the only value if bits_max is not set.",
                                           false, 10, "Expects a float.");
    TCLAP::ValueArg<double> bits_max_arg("", "bits_max",
                                         "The max value for the bits per positive element",
                                         false, 100, "Expects a float.");
    TCLAP::ValueArg<double> bits_step_arg("", "bits_step",
                                          "The step size over which [bits, bits_max) will be explored.",
                                          false, 1000, "Expects a float.");
    TCLAP::ValueArg<uint64> neg_universe_size_arg("", "neg_universe_size",
                                                  "The starting value for the negative universe size, and the only value if neg_universe_size_max is not set.",
                                                  false, 10000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> neg_universe_size_max_arg("", "neg_universe_size_max",
                                                      "The max value for the negative universe size.",
                                                      false, 10000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> neg_universe_size_ratio_arg("", "neg_universe_size_ratio",
                                                        "The ratio between steps over which [neg_universe_size, neg_universe_size_max) will be explored.",
                                                        false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint64> max_known_negatives_arg("", "max_known_negatives",
                                                    "The maximum number of known negatives available to the stacked filter.",
                                                    false, 5000000, "Expects a uint64.");
    TCLAP::ValueArg<uint64> max_known_negatives_max_arg("", "max_known_negatives_max",
                                                        "The max value for the max number of negatives it can use.",
                                                        false, 100000000000, "Expects a uint64.");
    TCLAP::ValueArg<double> max_known_negatives_ratio_arg("", "max_known_negatives_ratio",
                                                          "The ratio between steps over which [max_known_negatives, max_known_negatives_max) will be explored.",
                                                          false, 100000000, "Expects a double.");
    TCLAP::ValueArg<uint32> num_reps_arg("", "num_reps",
                                         "The number of repetitions per set of parameters.",
                                         false, 25, "Expects a uint32.");
    TCLAP::ValueArg<std::string> file_path_arg("", "path",
                                               "The path to the output file, relative or absolute.",
                                               false, "Data/TestData.csv", "Expects a string.");
    cmdLine.add(zipf_arg);
    cmdLine.add(zipf_max_arg);
    cmdLine.add(zipf_step_arg);
    cmdLine.add(bits_begin_arg);
    cmdLine.add(bits_max_arg);
    cmdLine.add(bits_step_arg);
    cmdLine.add(neg_universe_size_arg);
    cmdLine.add(neg_universe_size_max_arg);
    cmdLine.add(neg_universe_size_ratio_arg);
    cmdLine.add(max_known_negatives_arg);
    cmdLine.add(max_known_negatives_max_arg);
    cmdLine.add(max_known_negatives_ratio_arg);
    cmdLine.add(num_reps_arg);
    cmdLine.add(file_path_arg);
    cmdLine.parse(arg_num, args);

    zipf_begin = zipf_arg.getValue();
    zipf_max = zipf_max_arg.getValue();
    zipf_step = zipf_step_arg.getValue();
    bits_begin = bits_begin_arg.getValue();
    bits_max = bits_max_arg.getValue();
    bits_step = bits_step_arg.getValue();
    neg_universe_size = neg_universe_size_arg.getValue();
    neg_universe_size_max = neg_universe_size_max_arg.getValue();
    neg_universe_size_ratio = neg_universe_size_ratio_arg.getValue();
    max_known_negatives = max_known_negatives_arg.getValue();
    max_known_negatives_max = max_known_negatives_max_arg.getValue();
    max_known_negatives_ratio = max_known_negatives_ratio_arg.getValue();
    num_reps = num_reps_arg.getValue();
    file_path = file_path_arg.getValue();
}

int main(int arg_num, char **args) {
    TCLAP::CmdLine cmd("", '=', "0.1");
    double zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step, neg_universe_size_ratio, max_known_negatives_ratio;
    uint64 neg_universe_size_begin, neg_universe_size_max, max_known_negatives_begin, max_known_negatives_max;
    uint32 num_reps;
    std::string file_path;
    set_flags(cmd, zipf_begin, zipf_max, zipf_step, bits_begin, bits_max, bits_step,
              neg_universe_size_begin, neg_universe_size_max,
              neg_universe_size_ratio, max_known_negatives_begin, max_known_negatives_max,
              max_known_negatives_ratio,
              num_reps, file_path,
              arg_num, args
    );
    std::ofstream file_stream;
    file_stream.open(file_path);
    file_stream
            << "Number Of Positive Elements,Sample Size,Negative Universe Size,Number of Known Negatives Available,"
               "Number of Known Negatives Chosen,Psi,Bits Available,Used Bits,Total "
               "FPR\n";
    const uint64 negative_sample_size = 1000000;
    std::vector<StringElement> positives = get_positive_urls();
    std::vector<double> cdf, pmf;
    std::tie(cdf, pmf) = get_cdf_and_pmf();

    for (double zipf_parameter = zipf_begin; zipf_parameter < zipf_max; zipf_parameter += zipf_step) {
        if (zipf_parameter <= 1.01 && zipf_parameter >= .99) continue;
        for (double neg_universe_size = neg_universe_size_begin;
             neg_universe_size < neg_universe_size_max; neg_universe_size *= neg_universe_size_ratio) {
            std::vector<StringElement> negatives = get_negative_urls();
            for (uint64 max_known_negatives = max_known_negatives_begin;
                 max_known_negatives <=
                 max_known_negatives_max; max_known_negatives *= max_known_negatives_ratio) {
                for (double bits_per_positive_element = bits_begin;
                     bits_per_positive_element <= bits_max; bits_per_positive_element += bits_step) {
                    GenerateDataForOneRun(file_stream,
                                          neg_universe_size,
                                          bits_per_positive_element, num_reps,
                                          max_known_negatives,
                                          negative_sample_size, positives, negatives,
                                          cdf, pmf);
                    file_stream.flush();
                }
            }
        }
    }
    file_stream.close();
}

#pragma clang diagnostic pop