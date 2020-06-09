import sys
import os
import pickle
import argparse
import tqdm

import fairseq
import torch

# print out constraints
def print_c(d, dirname, vocab):
    tgt = d['tgt']
    with open(os.path.join(dirname, 'tgt.txt'), 'w') as fout:
        fout.write(' '.join([vocab[item.item()] for item in tgt[0]]))

    names = ['unigram', 'bigram', 'trigram', '4gram', '5gram']
    with open(os.path.join(dirname, 'constraints.txt'), 'w') as fout:
        for i, name in enumerate(names):
            if name in d:
                fout.write('='*50)
                if i+1 < len(names) and names[i+1] in d:
                    fout.write(f'{name.capitalize()} Constraints (Iteration {i+1})')
                else:
                    fout.write(f'{name.capitalize()} Viterbi Result (Iteration {i+1})')
                fout.write('='*50 + '\n')
                ngram_tokens = d[name]['tokens']
                if ngram_tokens.dim() == 1:
                    ngram_tokens = ngram_tokens.unsqueeze(0).unsqueeze(0)
                elif ngram_tokens.dim() == 2:
                    ngram_tokens = ngram_tokens.unsqueeze(-1)
                array = []
                for i in range(ngram_tokens.size(0)):
                    a = []
                    for k in range(ngram_tokens.size(1)):
                        a.append(' '.join([vocab[item] for item in ngram_tokens[i][k]]))
                    array.append(a)
                
                array2 = []
                for i in range(ngram_tokens.size(1)):
                    a = []
                    for j in range(len(array)):
                        a.append(array[j][i])
                    array2.append(a)
                for idx, items in enumerate(array2):
                    fout.write(f'{idx}: ' + '  & '.join(items)+ '\n')


def parse_args(args):
    parser = argparse.ArgumentParser(description='visualize_3d')
    parser.add_argument('--dump-vis-path', type=str, required=True,
                        help=('Dump data for visualization purposes'))
    parser.add_argument('--output-dir', type=str, required=True,
                        help=('Output directory.'))
    return parser.parse_args(args)

def main(args):
    parameters = parse_args(args)
    checkpoint = pickle.load(open(parameters.dump_vis_path, 'rb'))
    vocab = checkpoint['vocab']
    data = checkpoint['data']

    print (f'Processing {len(data)} sentences')
    for d in tqdm.tqdm(data):
        idx = d['id']
        dirname = os.path.join(parameters.output_dir, str(idx.item()))
        os.makedirs(dirname, exist_ok=True)

        print_c(d, dirname, vocab)

if __name__ == '__main__':
    main(sys.argv[1:])
