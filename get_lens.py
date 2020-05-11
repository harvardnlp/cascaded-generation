import sys, os
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
if len(sys.argv) != 4:
    print ('Usage: python %s <input1> <input2> <output>'%sys.argv[0])
    sys.exit(1)

xs = []
ys = []
with open(sys.argv[1]) as fin1:
    with open(sys.argv[2]) as fin2:
        for line1, line2 in zip(fin1, fin2):
            xs.append(len(line1.strip().split()))
            ys.append(len(line2.strip().split()))

        plt.plot(xs, ys, 'r.')

        plt.savefig(sys.argv[3])

        xs = np.array(xs)
        ys = np.array(ys)

        slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
        print (slope, intercept)
