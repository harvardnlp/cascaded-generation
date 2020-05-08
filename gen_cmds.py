import os 


rounds = range(2, 6)
topks = [16, 32, 64, 128, 256]
Ds = [0, 1, 2, 3]


dataset = 'iwslt'
dataset = 'iwslt-valid'
dataset = 'iwslt-distill-valid'


data_path_dict = {
        'iwslt': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',
        'iwslt-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',
        'iwslt-distill-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en-distill'
        }

model_path_dict = {
        'iwslt': '../fairseq_iwslt/checkpoints_null2_retrain/checkpoint_best.pt',
        'iwslt-valid': '../fairseq_iwslt/checkpoints_null2_retrain/checkpoint_best.pt',
        'iwslt-distill-valid': '../fairseq_iwslt/checkpoints_null2_distill/checkpoint_best.pt'
        }


gpu = 0
path = model_path_dict[dataset]
data_path = data_path_dict[dataset]

log_dir = f'/n/rush_lab/users/y/cascade_logs/{dataset}/'
os.makedirs(log_dir, exist_ok=True)

fouts = []
for i in range(4):
    fout = open(os.path.join(log_dir, f'{i}_cmd.sh'), 'w')
    fouts.append(fout)
for topk in topks:
    for round in rounds:
        for D in Ds:
            log_file_1 = os.path.join(log_dir, f'topk{topk}_D{D}_rounds{round}_speed.txt')
            log_file_2 = os.path.join(log_dir, f'topk{topk}_D{D}_rounds{round}_score.txt')

            cmd1 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --beam {topk} --rounds {round} --remove-bpe --D {D} > {log_file_1} 2>&1'
            cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --beam {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} > {log_file_2} 2>&1'
            if 'valid' in dataset:
                cmd1 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --beam {topk} --rounds {round} --remove-bpe --D {D} --gen-subset valid --maxval 3000 > {log_file_1} 2>&1'
                cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --beam {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} --gen-subset valid --maxval 3000 > {log_file_2} 2>&1'
            fouts[gpu].write(cmd1+'\n')
            fouts[gpu].write(cmd2+'\n')
            gpu += 1
            gpu = gpu % 4
for i in range(4):
    fouts[i].close()
