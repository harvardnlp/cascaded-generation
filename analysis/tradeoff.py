import sys, os
import re, subprocess

dataset = 2

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('Usage: python %s <input> <output>'%sys.argv[0])
        sys.exit(1)


with open(sys.argv[1]) as fin:
    with open(sys.argv[2], 'w') as fout:
        for line in fin:
            items = line.strip().split('&')
            name = items[0]
            m = re.match(r'.*?topk\s*=\s*(\d+).*?', name)
            assert m, name
            topk = int(m.group(1))
            if topk > 128:
                continue
            #print (topk)
            m = re.match(r'.*?rounds\s*=\s*(\d+).*?', name)
            assert m, name
            rounds = int(m.group(1))
            #print (rounds)
            m = re.match(r'.*?D\s*=\s*(\d+).*?', name)
            assert m, name
            D = int(m.group(1))

            idx = 2 + 2*dataset
            item1 = items[idx]
            m = re.match(r'\s*([\d\.]+).*?', item1)
            if not m:
                continue
            bleu = float(m.group(1))
            item2 = items[idx+1]
            m = re.match(r'\s*([\d\.]+).*?', item2)
            assert m, item2
            latency = m.group(1)
            latency = float(latency)

            fout.write(f'{topk}\t{rounds}\t{D}\t{bleu}\t{latency}\n')

            #print (D)

            #output_items = []
            #output_items.append(name)
            #output_items.append(items[1])

            #output_items[-1] = output_items[-1] + '\\\\'
            #fout.write('&'.join(output_items) + '\n')
