from DeepBloomLib.BloomFilter import BloomFilter
import math
import random
from DeepBloomLib.utils import *
import murmurhash as mmh3


class SandwichedLearnedBloom(object):
    def __init__(self, model, data, desiredSizeBitsPerElement, modelSize, train_model=False):
        self.model = model
        self.val_negatives = None
        self.threshold = None
        self.modelSize = modelSize * 8
        self.desiredSizeBitsPerElement = desiredSizeBitsPerElement
        if (train_model):
            self.fit(data)
        else:
            (s1, s2) = split_negatives(data)
            self.val_negatives = s2
        self.hasInitialFilter = False

    def check(self, item):
        if self.hasInitialFilter and self.bloom_filter1.check(item) == False:
            return False
        if self.model.predict(item) > self.threshold:
            return True
        return self.bloom_filter2.check(item)

    def calculateOptimalBackupSize(self):
        bf_scale_factor = 0.5 ** (math.log(2))
        log_term_denom = (1 - self.Fp_rate_learned) * ((1 / self.Fn_rate_learned) - 1)
        log_term = self.Fp_rate_learned / (log_term_denom)
        ### NOTE THIS IS BITS PER BACKUP FILTER
        self.bits_backup_opt = max(1, self.Fn_rate_learned * (math.log(log_term) / math.log(bf_scale_factor)))
        #print("Number of optimal bits backup filter: {}".format(self.bits_backup_opt))

    def createFilterGivenModelFPR(self, data, preds, predictions, modelFPR):
        fp_index = math.ceil((len(self.val_negatives) * (1 - modelFPR)))
        self.threshold = predictions[max(min(fp_index, len(predictions)-1), 0)]
        self.Fp_rate_learned = modelFPR
        false_negatives = []
        for i in range(len(data.positives)):
            if preds[i] <= self.threshold:
                false_negatives.append(data.positives[i])
        self.num_false_negatives = len(false_negatives)
        self.Fn_rate_learned = self.num_false_negatives / len(data.positives)
        print("Model FPR: {}, Model FNR: {}".format(self.Fp_rate_learned, self.Fn_rate_learned))

        self.calculateOptimalBackupSize()
        totalSize = self.desiredSizeBitsPerElement * len(data.positives)
        assert(totalSize > self.modelSize)
        leftoverSize = totalSize - self.modelSize
        leftoverSizeInBitsPerElement = leftoverSize / len(data.positives)
        print("leftoverSizeInBitsPerElement: {}".format(leftoverSizeInBitsPerElement))
        print("num negatives for B2: {}".format(self.num_false_negatives))
        if (self.bits_backup_opt < leftoverSizeInBitsPerElement):
            self.hasInitialFilter = True
            self.bloom_filter1 = BloomFilter.initializeBloomFromSize(len(data.positives), len(data.positives) * (leftoverSizeInBitsPerElement - self.bits_backup_opt), string_digest)
            self.bloom_filter2 = BloomFilter.initializeBloomFromSize(self.num_false_negatives, len(data.positives) * self.bits_backup_opt, string_digest)
            for positive in data.positives:
                self.bloom_filter1.add(positive)     
            for fn in false_negatives:
                self.bloom_filter2.add(fn)
        else:
            self.hasInitialFilter = False
            self.bloom_filter1 = None
            self.bloom_filter2 = BloomFilter.initializeBloomFromSize(len(data.positives), len(data.positives) * leftoverSizeInBitsPerElement, string_digest)
            for fn in false_negatives:
                self.bloom_filter2.add(fn)

    def create_bloom_filter(self, data):
        preds = self.model.predicts(data.positives)
        predictions = self.model.predicts(self.val_negatives)
        predictions.sort()
        traditionalBF_FPR = ((0.5) ** math.log(2)) ** (self.desiredSizeBitsPerElement)
        print("Traditional BF FPR: {}".format(traditionalBF_FPR))
        ModelFPRs = np.logspace(np.log10(traditionalBF_FPR / 20), np.log10(traditionalBF_FPR), 10)
        bestFPR = 1
        for modelFPR in ModelFPRs:
            self.createFilterGivenModelFPR(data, preds, predictions, modelFPR)
            if(self.getExpectedFalsePositiveRate() < bestFPR):
                bestFPR = self.getExpectedFalsePositiveRate()
                bestModelFPR = modelFPR
            elif(self.getExpectedFalsePositiveRate() > bestFPR * 1.5):
                break
            print("MODEL FPR: {}, Filter FPR: {}".format(modelFPR, self.getExpectedFalsePositiveRate()))
        self.createFilterGivenModelFPR(data,preds,predictions, bestModelFPR)
        print("BEST MODEL FPR: {}, Filter FPR: {}".format(bestModelFPR, self.getExpectedFalsePositiveRateWithPrint()))

    def getExpectedFalsePositiveRateWithPrint(self):
        if self.hasInitialFilter == True:
            print("FPR B1: {}, FPR Learned: {}, FPR B3: {}".format(self.bloom_filter1.getFPR(), self.Fp_rate_learned, self.bloom_filter2.getFPR()))
            self.fp_rate = self.bloom_filter1.getFPR() * (self.Fp_rate_learned + (1 - self.Fp_rate_learned) * self.bloom_filter2.getFPR())
            return self.bloom_filter1.getFPR() * (self.Fp_rate_learned + (1 - self.Fp_rate_learned) * self.bloom_filter2.getFPR())
        else:
            print("FPR Learned: {}, FPR B3: {}".format(self.Fp_rate_learned, self.bloom_filter2.getFPR()))
            return (self.Fp_rate_learned + (1 - self.Fp_rate_learned) * self.bloom_filter2.getFPR())
        
    def getExpectedFalsePositiveRate(self):
        if self.hasInitialFilter == True:
            self.fp_rate = self.bloom_filter1.getFPR() * (self.Fp_rate_learned + (1 - self.Fp_rate_learned) * self.bloom_filter2.getFPR())
            return self.bloom_filter1.getFPR() * (self.Fp_rate_learned + (1 - self.Fp_rate_learned) * self.bloom_filter2.getFPR())
        else:
            return (self.Fp_rate_learned + (1 - self.Fp_rate_learned) * self.bloom_filter2.getFPR())

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
