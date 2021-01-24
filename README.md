# StackedAlternatingAMQs

## Overview

The Stacked Filter is an adaptation of traditional filtering structures, such as Bloom
or Cuckoo Filters, which uses additional workload information to increase the space
efficiency and lower the false positive rate. It does this by using these 
traditional filters as layers which alternate storing positive or negative elements.
Positives are elements known to be in the set represented by the filter while
negatives are elements known to not be in the set represented by the filter.

For details about the algorithm and citations, please see 
*[Stacked Filters: Learning to Filter by Structure](https://stratos.seas.harvard.edu/files/stratos/files/stackedfilters_pvldb21.pdf)*
in proceedings of VLDB 2021 by Brian Hentschel, Kyle Deeds, and Stratos Idreos.

## Stacked Filter API

There are three main functions in the Stacked Filter API: bulk construction, LookupElement, 
and InsertPositiveElement.

    StackedFilter(const size_t total_size,
                  const std::vector<element_type> &positives,
                  const std::vector<element_type> &negatives,
                  const std::vector<double> &cdf,
                  const double insert_capacity = 0);

    bool LookupElement(const element_type element);

    void InsertPositiveElement(const element_type element);

Bulk construction takes in a size constraint, a list of positive elements, a 
list of negative elements, the partial cdf of the query frequency for those 
negative elements, and an insert capacity. The second to last two parameters have a few specifications: 
the negative elements should be in descending order by their estimated frequencies,
and the cdf should be an increasing sum of frequencies (0<=F(x)<=1) matching
the order of the negative list, see 
[Rank-Frequency Distribution](https://en.wikipedia.org/wiki/Rank-size_distribution#:~:text=Rank%2Dsize%20distribution%20is%20the,in%20decreasing%20order%20of%20size.&text=This%20is%20also%20known%20as,city%20size%20or%20word%20frequency).https://en.wikipedia.org/wiki/Rank-size_distribution#:~:text=Rank%2Dsize%20distribution%20is%20the,in%20decreasing%20order%20of%20size.&text=This%20is%20also%20known%20as,city%20size%20or%20word%20frequency).
Lastly, insert_capacity represents the amount of extra space that should be allotted 
in the positive layers in order to handle future inserts. This is given in terms of the 
number of future elements divided by the number of starting elements.

LookupElement returns whether an element is present in the set with some small
probability of a false positive which depends on the size.

InsertPositiveElement allows for the insertion of elements into the positive set. Note
that negative elements cannot be inserted after construction. Due to the structure
of the filter, it is generally not possible to add negative data after construction, 
but see include/StackedFilter/AdaptiveStackedBF.h for an alternative method to handle 
frequently changing negative sets.

## Repository Structure

`include/StackedFilter`: all StackedFilter data structure code, in particular
the file StackedFilter.h which implements the primary Stacked Filter logic.

`include/Optimization`: the optimization logic which translates a space budget,
number of positives, and negatives CDF into a Stacked Filter design.

`include/(BloomFilter|QuotientFilter|CuckooFilter)`: versions of traditional
filters which can be used as layers in a stacked filter. The quotient filter code is 
derived from [A General Purpose Counting Filter: Making Every Bit Count](https://github.com/splatlab/cqf),
while the Cuckoo Filter is from [Cuckoo Filters Practically Better Than Bloom](http://www.cs.cmu.edu/~binfan/papers/conext14_cuckoofilter.pdf).
The Bloom Filter was simply re-implemented.

`Main/`: main files which run various experiments from the original paper. In addition, 
a file `example.exe` which demonstrates a very simple usage of Stacked Filters.

## Build

This repository uses CMake to manage its build process.




