from DeepBloomLib.BloomFilter import BloomFilter
import math
import random
from DeepBloomLib.utils import *
import murmurhash as mmh3


class DeepBloom(object):
    def __init__(self, model, data, fp_rate, train_model=True):
        self.model = model
        self.val_negatives = None
        self.threshold = None
        self.fp_rate = float(fp_rate)
        if (train_model):
            self.fit(data)
        else:
            (s1, s2) = split_negatives(data)
            self.val_negatives = s2

    def check(self, item):
        if self.model.predict(item) > self.threshold:
            return True
        return self.bloom_filter.check(item)

    def create_bloom_filter(self, data):
        print("Calculating Threshold")
        ## We want a threshold such that at most s2.size * fp_rate/2 elements
        ## are greater than threshold.
        fp_index = math.ceil((len(self.val_negatives) * (1 - self.fp_rate / 2)))
        predictions = self.model.predicts(self.val_negatives)
        predictions.sort()
        self.threshold = predictions[fp_index]

        print("Creating bloom filter")
        false_negatives = []
        preds = self.model.predicts(data.positives)
        for i in range(len(data.positives)):
            if preds[i] <= self.threshold:
                false_negatives.append(data.positives[i])
        self.num_false_negatives = len(false_negatives)
        print("Number of false negatives at bloom time", len(false_negatives))
        self.bloom_filter = BloomFilter(
            len(false_negatives),
            self.fp_rate / 2,
            string_digest
        )
        for fn in false_negatives:
            self.bloom_filter.add(fn)
        print("Created bloom filter")

    def create_fake_bloom_filter(self, data):
        print("Calculating Threshold")
        (s1, s2) = split_negatives(data)
        self.val_negatives = s2
        ## We want a threshold such that at most s2.size * fp_rate/2 elements
        ## are greater than threshold.
        fp_index = math.ceil((len(self.val_negatives) * (1 - self.fp_rate / 2)))
        predictions = self.model.predicts(self.val_negatives)
        predictions.sort()
        self.threshold = predictions[fp_index]

        print("Creating bloom filter")
        false_negatives = list(filter(lambda x: x <= self.threshold, self.model.predicts(data.positives)))
        print("Number of false negatives at bloom time", len(false_negatives))
        self.bloom_filter = BloomFilter(
            len(false_negatives),
            self.fp_rate / 2,
            string_digest
        )
        print("Created bloom filter")

    def fit(self, data):
        ## Split negative data into subgroups.
        (s1, s2) = split_negatives(data)
        print("Training model with train, dev, positives", len(s1), len(s2), len(data.positives))

        ## Shuffle together subset of negatives and positives.
        ## Then, train the model on this data.
        shuffled = shuffle_for_training(s1, data.positives)
        self.model.fit(shuffled[0], shuffled[1])
        print("Done fitting")

        ## Store the test negatives for later use in generating a threshold
        self.val_negatives = s2
