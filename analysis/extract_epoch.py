import sys, os
import re, subprocess
import glob

distilled = False
base_dir = '/n/rush_lab/users/y/validation_logs/'
dirs = ['wmt-en-ro-epoch', 'wmt-ro-en-epoch']
dirs_val = ['wmt-en-ro-epoch-valid', 'wmt-ro-en-epoch-valid']
dirs = ['wmt-en-de-distill-240k-epoch', 'wmt-de-en-distill-240k-epoch']
dirs_val = ['wmt-en-de-distill-240k-epoch-valid', 'wmt-de-en-distill-240k-epoch-valid']
if distilled:
    dirs = [item + '-distill' for item in dirs]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('Usage: python %s <output>'%sys.argv[0])
        sys.exit(1)

def get_topk_rounds_D(d):
    print (d)
    filenames = glob.glob(os.path.join(d, '*'))
    for filename in filenames:
        m = re.match(r'epoch(\d+)_topk(\d+)_D(\d+)_rounds(\d+)_score.txt', os.path.basename(filename))
        if not m:
            continue
        topk = m.group(2)
        D = m.group(3)
        rounds = m.group(4)
        return topk, rounds, D

for d, d_val in zip(dirs, dirs_val):
    with open(sys.argv[1]+d, 'w') as fout:
        scores = []
        bleus = []
        scores_val = []
        bleus_val = []
        d = os.path.join(base_dir, d)
        d_val = os.path.join(base_dir, d_val)
        topk, rounds, D = get_topk_rounds_D(d)
        topk_val, rounds_val, D_val = get_topk_rounds_D(d_val)
    
        epoch = 0
        while True:
            epoch += 1
            filename = f'epoch{epoch}_topk{topk}_D{D}_rounds{rounds}_score.txt'
            log_file = os.path.join(d, filename)
            filename_val = f'epoch{epoch}_topk{topk_val}_D{D_val}_rounds{rounds_val}_score.txt'
            log_file_val = os.path.join(d_val, filename_val)
    
            if (not os.path.exists(log_file)) or (not os.path.exists(log_file_val)):
                break
    
            lines = open(log_file).readlines()[-2:]
            m = re.match(r'.*?BLEU4\s*=\s*([\d\.]+).*?', lines[1])
            if not m:
                break
            bleu = m.group(1)
            bleu = float(bleu)
    
            cmd = f'grep ^P {log_file} | cut -f2- > tmp.txt && python avg.py tmp.txt'
            output = subprocess.check_output(cmd, shell=True)
            score = float(output.strip())
    
            lines = open(log_file_val).readlines()[-2:]
            m = re.match(r'.*?BLEU4\s*=\s*([\d\.]+).*?', lines[1])
            if not m:
                break
            bleu_val = m.group(1)
            bleu_val = float(bleu_val)
    
            cmd = f'grep ^P {log_file_val} | cut -f2- > tmp.txt && python avg.py tmp.txt'
            output = subprocess.check_output(cmd, shell=True)
            score_val = float(output.strip())
    
            scores.append(score)
            bleus.append(bleu)
            scores_val.append(score_val)
            bleus_val.append(bleu_val)
            fout.write(f'{bleu}\t{bleu_val}\t{score}\t{score_val}\n')
