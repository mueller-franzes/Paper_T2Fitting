
import numpy as np
import scipy.stats as stats


class UniformExpGen(stats.rv_continuous):
    "Unify Distribution [k1, k2] + Exponential Decay for >k2"

    def _pdf(self, x):
        # def _pdf(self, x, k1, k2):
        # k1 = k1-1 # Workaround k1 must be !=0 , thus k1=1 correspond to k1=0
        k1 = 0
        k2 = 500

        diff = k2 - k1
        if x < k1:
            return 0
        elif x < k2:
            return 1 / (2 * diff)
        else:
            return np.exp(1 - x / k2) / (2 * diff)

    def _cdf(self, x):
        # def _cdf(self, x, k1, k2):
        # k1 = k1-1 # Workaround k1 must be !=0 , thus k1=1 correspond to k1=0
        k1 = 0
        k2 = 500
        diff = k2 - k1
        if x < k1:
            return 0
        elif x < k2:
            return x / (2 * diff)
        else:
            return 1 - 0.5 * np.exp(1 - x / k2)


uniform_exp = UniformExpGen()



