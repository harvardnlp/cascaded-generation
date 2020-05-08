import sys, os
import re, subprocess

distilled = True
valid = False
base_dir = '/n/rush_lab/users/y/cascade_logs/'
dirs = ['wmt-en-de-240k', 'wmt-de-en-240k', 'iwslt']
if distilled:
    dirs = [item + '-distill' for item in dirs]
    dirs = ['wmt-en-de-distill-240k', 'wmt-de-en-distill-240k', 'iwslt-distill']
if valid:
    dirs = [item + '-valid' for item in dirs]

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
            #print (topk)
            m = re.match(r'.*?rounds\s*=\s*(\d+).*?', name)
            assert m, name
            rounds = int(m.group(1))
            #print (rounds)
            m = re.match(r'.*?D\s*=\s*(\d+).*?', name)
            assert m, name
            D = int(m.group(1))
            #print (D)

            output_items = []
            output_items.append(name)
            output_items.append(items[1])

            for i in range(3):
                log_dir = base_dir + dirs[i]
                log_file_1 = os.path.join(log_dir, f'topk{topk}_D{D}_rounds{rounds}_speed.txt')
                log_file_2 = os.path.join(log_dir, f'topk{topk}_D{D}_rounds{rounds}_score.txt')

                if os.path.exists(log_file_1) and os.path.exists(log_file_2):
                    lines = open(log_file_1).readlines()[-2:]
                    m = re.match(r'.*?Latency\s*([\d\.]+).*?', lines[0])
                    if not m:
                        output_items.append(' \t')
                        output_items.append(' \t')
                        continue
                    latency = float(m.group(1))
                    m = re.match(r'.*?BLEU4\s*=\s*([\d\.]+).*?', lines[1])
                    assert m, lines
                    bleu = m.group(1)

                    cmd = f'grep ^P {log_file_2} | cut -f2- > tmp.txt && python avg.py tmp.txt'
                    output = subprocess.check_output(cmd, shell=True)
                    score = float(output.strip())
                    score = round(score, 2)
                    latency = round(latency, 2)

                    print (bleu, score, latency)
                    output_items.append(f' {bleu} ({score})\t')
                    output_items.append(f' {latency}ms\t')
                else:
                    print (log_file_1)
                    output_items.append(' \t')
                    output_items.append(' \t')
            output_items[-1] = output_items[-1] + '\\\\'
            fout.write('&'.join(output_items) + '\n')
