import sys, os
import torch


import seaborn
import matplotlib.pyplot as plt
seaborn.set(font='serif', font_scale=2.0)
seaborn.set_context("paper")
seaborn.set_style("whitegrid")
seaborn.set(font_scale=1.5)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('Usage: python %s <input> <output>'%sys.argv[1])
        sys.exit(1)

    xs, ys, ds = [], [], []
    with open(sys.argv[1]) as fin:
        for line in fin:
            items = line.strip().split('\t')
            bleu = items[0]
            bleu_val = items[1]
            xs.append(float(bleu))
            ys.append(float(bleu_val))
    if 'ro-en' in sys.argv[1]:
        max_len = 105

    xs = xs[:max_len]
    ys = ys[:max_len]
    plt.plot(range(len(xs)), xs, 'r-', range(len(ys)), ys, 'b--')
    plt.ylabel('BLEU')
    plt.xlabel('epoch')
    plt.legend(['test', 'val'], loc='upper right')
    plt.tight_layout()
    #fig.subplots_adjust(bottom=0.3)

    plt.savefig(sys.argv[2])
