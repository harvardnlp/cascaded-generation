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
    #assert 'iwslt' in sys.argv[1]
    with open(sys.argv[1]) as fin:
        for line in fin:
            items = line.strip().split('\t')
            topk = int(items[0])
            rounds = int(items[1])
            D = int(items[2])
            bleu = float(items[3])
            latency = float(items[4])

            xs.append(1./latency)
            ys.append(bleu)
            ds.append(D)
    xs = torch.Tensor(xs)
    #xs = 229.76*xs
    #xs = 318.85*xs # wmt en-de
    #330.55872330
    xs = 330.55872330*xs # wmt en-de
    ys = torch.Tensor(ys)
    ds = torch.LongTensor(ds)


    x = xs[ds.eq(0)].tolist()
    y = ys[ds.eq(0)].tolist()
    plt.plot(x, y, 'r*')
    x = xs[ds.eq(1)].tolist()
    y = ys[ds.eq(1)].tolist()
    plt.plot(x, y, 'gD')
    x = xs[ds.eq(2)].tolist()
    y = ys[ds.eq(2)].tolist()
    plt.plot(x, y, 'bo')
    x = xs[ds.eq(3)].tolist()
    y = ys[ds.eq(3)].tolist()
    plt.plot(x, y, 'k.')
    plt.ylabel('BLEU')
    plt.xlabel('Speedup')
    #plt.ylim(32, 35)
    plt.legend(['D=0', 'D=1', 'D=2', 'D=3'], loc='upper right')
    plt.tight_layout()
    #fig.subplots_adjust(bottom=0.3)

    plt.savefig(sys.argv[2])
