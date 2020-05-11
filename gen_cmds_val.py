import os 
import glob, re

rounds = range(2, 6)
topks = [16, 32, 64, 128]
Ds = [0, 1, 2, 3]

#topks = [128,]
Ds = [3, ]

dataset = 'wmt-de-en-valid'
dataset = 'wmt-en-de-distill-valid'
dataset = 'wmt-de-en-distill'
dataset = 'wmt-de-en-distill-valid'
dataset = 'wmt-en-de-distill-times2'
dataset = 'wmt-en-de-distill-times4'
dataset = 'wmt-en-de'
dataset = 'wmt-en-de-dgx'
dataset = 'wmt-en-de-valid'
dataset = 'table3-iwslt-valid'
dataset = 'wmt-en-de-lambda'
dataset = 'wmt-en-de-last'
dataset = 'wmt-en-de-240k'
dataset = 'wmt-en-de-lambda-last'
dataset = 'wmt-en-de-240k'
dataset = 'wmt-de-en-240k'
dataset = 'wmt-en-de-distill-240k'
dataset = 'wmt-de-en-distill-240k'
dataset = 'iwslt-distill'
dataset = 'wmt-en-ro-epoch-valid'
dataset = 'wmt-ro-en-epoch-valid'

dataset = 'wmt-en-ro-epoch'
dataset = 'wmt-ro-en-epoch'
dataset = 'wmt-en-ro-epoch'
dataset = 'wmt-en-de-distill-240k-epoch-valid'
dataset = 'wmt-de-en-distill-240k-epoch-valid'

dataset = 'wmt-en-de-distill-240k-epoch'
dataset = 'wmt-de-en-distill-240k-epoch'

is_valid = True
datasets = ['wmt-en-ro-distill-epoch-ab', 'wmt-ro-en-distill-epoch-ab', 'wmt-en-ro-epoch-ab', 'wmt-ro-en-epoch-ab', 'wmt-en-de-distill-epoch-ab', 'wmt-de-en-distill-epoch-ab', 'wmt-en-de-epoch-ab', 'wmt-de-en-epoch-ab', 'iwslt-distill-epoch-ab', 'iwslt-epoch-ab']
datasets = ['wmt-en-ro-distill-epoch-ab', 'wmt-ro-en-distill-epoch-ab', 'wmt-en-ro-epoch-ab', 'wmt-ro-en-epoch-ab', 'wmt-en-de-distill-epoch-ab', 'wmt-de-en-distill-epoch-ab', 'wmt-en-de-epoch-ab', 'wmt-de-en-epoch-ab']
datasets = ['wmt-en-de-epoch-ab']

if is_valid:
    datasets = [item + '-valid' for item in datasets]

max_len_ab_dict = {
        'wmt-en-ro-distill-epoch-ab': [0.9883130097474603, 0.9633488906317602],
        'wmt-ro-en-distill-epoch-ab': [0.9248859339993715, 0.8017646630157031],
        'wmt-en-ro-epoch-ab': [0.996921062198531, 1.0282739671455445],
        'wmt-ro-en-epoch-ab': [0.9247462305812968, 1.1175631354616158],
        'wmt-en-de-distill-epoch-ab': [0.9806108887946017, 0.8722537432612825],
        'wmt-de-en-distill-epoch-ab': [0.916756767932658, 0.9342969197602784],
        'wmt-en-de-epoch-ab': [0.9799950605387177, 1.2678756896665853],
        'wmt-de-en-epoch-ab': [0.9249639109649428, 1.4871392936978083],
        'iwslt-distill-epoch-ab': [0.8853894031751983, 0.9270677277453601],
        'iwslt-epoch-ab': [0.941281036889224, 0.8804326732522796],
}

data_path_dict = {
        'iwslt': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',
        'wmt-de-en': '../fairseq_vanilla/data-bin/wmt17_de_en_joint', 
        'wmt-de-en-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_joint', 
        'wmt-en-de-distill-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_distill_valid', 
        'wmt-de-en-distill': '../fairseq_vanilla/data-bin/wmt17_de_en_distill', 
        'wmt-de-en-distill-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_distill_valid', 
        'wmt-en-de-distill': '../fairseq_vanilla/data-bin/wmt17_en_de_distill', 
        'wmt-en-de-distill-times2': '../fairseq_vanilla/data-bin/wmt17_en_de_distill', 
        'wmt-en-de-distill-times4': '../fairseq_vanilla/data-bin/wmt17_en_de_distill', 
        'wmt-en-de': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'wmt-en-de-dgx': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'wmt-en-de-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'table3-iwslt-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',
        'wmt-en-de-last': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'wmt-en-de-lambda': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'wmt-en-de-240k': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'wmt-en-de-lambda-last': '../fairseq_vanilla/data-bin/wmt17_en_de_joint', 
        'wmt-en-de-distill-240k': '../fairseq_vanilla/data-bin/wmt17_en_de_distill', 
        'wmt-de-en-240k': '../fairseq_vanilla/data-bin/wmt17_de_en_joint', 
        'wmt-de-en-distill-240k': '../fairseq_vanilla/data-bin/wmt17_de_en_distill', 
        'iwslt-distill': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en-distill',
        'wmt-en-ro-epoch': '../fairseq_vanilla/data-bin/wmt16_en_ro_joint', 
        'wmt-ro-en-epoch': '../fairseq_vanilla/data-bin/wmt16_ro_en_joint', 
        'wmt-en-ro-epoch-valid': '../fairseq_vanilla/data-bin/wmt16_en_ro_joint', 
        'wmt-ro-en-epoch-valid': '../fairseq_vanilla/data-bin/wmt16_ro_en_joint', 
        'wmt-en-ro-epoch': '../fairseq_vanilla/data-bin/wmt16_en_ro_joint', 
        'wmt-ro-en-epoch': '../fairseq_vanilla/data-bin/wmt16_ro_en_joint', 
        'wmt-en-de-distill-240k-epoch-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_distill_valid', 
        'wmt-de-en-distill-240k-epoch-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_distill_valid', 
        'wmt-en-de-distill-240k-epoch': '../fairseq_vanilla/data-bin/wmt17_en_de_distill', 
        'wmt-de-en-distill-240k-epoch': '../fairseq_vanilla/data-bin/wmt17_de_en_distill',

        'wmt-en-ro-distill-epoch-ab': '../fairseq_vanilla/data-bin/wmt16_en_ro_distill',
        'wmt-ro-en-distill-epoch-ab': '../fairseq_vanilla/data-bin/wmt16_ro_en_distill',
        'wmt-en-ro-epoch-ab': '../fairseq_vanilla/data-bin/wmt16_en_ro_joint',
        'wmt-ro-en-epoch-ab': '../fairseq_vanilla/data-bin/wmt16_ro_en_joint',
        'wmt-en-de-distill-epoch-ab': '../fairseq_vanilla/data-bin/wmt17_en_de_distill_valid',
        'wmt-de-en-distill-epoch-ab': '../fairseq_vanilla/data-bin/wmt17_de_en_distill_valid',
        'wmt-en-de-epoch-ab': '../fairseq_vanilla/data-bin/wmt17_en_de_joint',
        'wmt-de-en-epoch-ab': '../fairseq_vanilla/data-bin/wmt17_de_en_joint',
        'iwslt-distill-epoch-ab': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en-distill-valid',
        'iwslt-epoch-ab': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',

        #'wmt-en-ro-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_en_ro_distill',
        #'wmt-ro-en-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_ro_en_distill',
        #'wmt-en-ro-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_en_ro_joint',
        #'wmt-ro-en-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_ro_en_joint',
        #'wmt-en-de-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_distill_valid',
        #'wmt-de-en-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_distill_valid',
        #'wmt-en-de-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_joint',
        #'wmt-de-en-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_joint',
        #'iwslt-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en-distill-valid',
        #'iwslt-epoch-ab-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',
}

model_path_dict = {
        'iwslt': '../fairseq_iwslt/checkpoints_null2_retrain/checkpoint_best.pt',
        'wmt-de-en': '/n/rush_lab/users/y/checkpoints/barrier/de-en/checkpoint_best.pt',
        'wmt-de-en-valid': '/n/rush_lab/users/y/checkpoints/barrier/de-en/checkpoint_best.pt',
        'wmt-en-de-distill-valid': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_best.pt',
        'wmt-de-en-distill': '/n/rush_lab/users/y/checkpoints/barrier/de-en-distill/checkpoint_best.pt',
        'wmt-de-en-distill-valid': '/n/rush_lab/users/y/checkpoints/barrier/de-en-distill/checkpoint_best.pt',
        'wmt-en-de-distill': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_best.pt',
        'wmt-en-de-distill-times2': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_best.pt',
        'wmt-en-de-distill-times4': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_best.pt',
        'wmt-en-de': '/n/rush_lab/users/y/checkpoints/barrier/en-de-retrain/checkpoint_best.pt',
        'wmt-en-de-dgx': '/n/rush_lab/users/y/checkpoints/barrier/en-de-retrain/checkpoint_best.pt',
        'wmt-en-de-valid': '/n/rush_lab/users/y/checkpoints/barrier/en-de-retrain/checkpoint_best.pt',
        'table3-iwslt-valid': '../fairseq_iwslt/checkpoints_null2_retrain/checkpoint_best.pt',
        'wmt-en-de-last': '/n/rush_lab/users/y/checkpoints/barrier/en-de-retrain/checkpoint_last.pt',
        'wmt-en-de-lambda': '/n/rush_lab/users/y/checkpoint_best.pt',
        'wmt-en-de-240k': '/n/rush_lab/users/y/checkpoints/barrier/en-de-retrain/checkpoint_68_240000.pt',
        'wmt-en-de-lambda-last': '/n/rush_lab/users/y/checkpoint_last.pt',
        'wmt-en-de-distill-240k': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_70_240000.pt',
        'wmt-de-en-240k': '/n/rush_lab/users/y/checkpoints/barrier/de-en/checkpoint_68_240000.pt',
        'wmt-de-en-distill-240k': '/n/rush_lab/users/y/checkpoints/barrier/de-en-distill-240k/checkpoint_70_240000.pt',
        'iwslt-distill': '../fairseq_iwslt/checkpoints_null2_distill/checkpoint_best.pt',
        'wmt-en-ro-epoch': '/n/rush_lab/users/y/checkpoints/barrier/en-ro-mlm-3gpu-largelr',
        'wmt-ro-en-epoch': '/n/rush_lab/users/y/checkpoints/barrier/ro-en-mlm-3gpu-largelr',
        'wmt-en-ro-epoch-valid': '/n/rush_lab/users/y/checkpoints/barrier/en-ro-mlm-3gpu-largelr',
        'wmt-ro-en-epoch-valid': '/n/rush_lab/users/y/checkpoints/barrier/ro-en-mlm-3gpu-largelr',
        'wmt-en-ro-epoch': '/n/rush_lab/users/y/checkpoints/barrier/en-ro-mlm-3gpu-largelr',
        'wmt-ro-en-epoch': '/n/rush_lab/users/y/checkpoints/barrier/ro-en-mlm-3gpu-largelr',
        'wmt-en-de-distill-240k-epoch-valid': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/',
        'wmt-de-en-distill-240k-epoch-valid': '/n/rush_lab/users/y/checkpoints/barrier/de-en-distill/',
        'wmt-en-de-distill-240k-epoch': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/',
        'wmt-de-en-distill-240k-epoch': '/n/rush_lab/users/y/checkpoints/barrier/de-en-distill/',


        'wmt-en-ro-distill-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/en-ro-mlm-3gpu-largelr-fix-distill/',
        'wmt-ro-en-distill-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/ro-en-mlm-3gpu-largelr-fix-distill/',
        'wmt-en-ro-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/en-ro-mlm-3gpu-largelr-fix',
        'wmt-ro-en-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/ro-en-mlm-3gpu-largelr-fix',
        'wmt-en-de-distill-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/en-de-distill/',
        'wmt-de-en-distill-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/de-en-distill/',
        'wmt-en-de-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/en-de-retrain/',
        'wmt-de-en-epoch-ab': '/n/rush_lab/users/y/checkpoints/barrier/de-en/',
        'iwslt-distill-epoch-ab': '../fairseq_iwslt/checkpoints_null2_distill/',
        'iwslt-epoch-ab': '../fairseq_iwslt/checkpoints_null2_retrain/',

        #'wmt-en-ro-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_en_ro_distill',
        #'wmt-ro-en-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_ro_en_distill',
        #'wmt-en-ro-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_en_ro_joint',
        #'wmt-ro-en-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt16_ro_en_joint',
        #'wmt-en-de-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_distill_valid',
        #'wmt-de-en-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_distill_valid',
        #'wmt-en-de-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_en_de_joint',
        #'wmt-de-en-epoch-ab-valid': '../fairseq_vanilla/data-bin/wmt17_de_en_joint',
        #'iwslt-distill-epoch-ab-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en-distill-valid',
        #'iwslt-epoch-ab-valid': '../fairseq_vanilla/data-bin/iwslt14.tokenized.de-en',
        }




for dataset in datasets:
    gpu = 0
    if 'valid' in dataset:
        path = model_path_dict[dataset[:-len('-valid')]]
        data_path = data_path_dict[dataset[:-len('-valid')]]
        max_len_a, max_len_b = max_len_ab_dict[dataset[:-len('-valid')]]
    else:
        path = model_path_dict[dataset]
        data_path = data_path_dict[dataset]
        max_len_a, max_len_b = max_len_ab_dict[dataset]
    
    log_dir = f'/n/rush_lab/users/y/validation_ab_logs/{dataset}/'
    os.makedirs(log_dir, exist_ok=True)
    
    fouts = []
    for i in range(4):
        fout = open(os.path.join(log_dir, f'{i}_cmd.sh'), 'w')
        fouts.append(fout)
    
    
    if 'epoch' in dataset:
        assert 'ab' in dataset
        d = path
        epoch = 0
        while True:
            epoch += 1
            path = os.path.join(d, f'checkpoint{epoch}.pt')
            if not os.path.exists(path):
                print (epoch)
                if epoch > 200:
                    break
                continue
            if 'valid' in dataset:
                topks = [16,]
            else:
                topks = [32,]
            rounds = [5,]
            Ds = [3,]
            for topk in topks:
                for round in rounds:
                    for D in Ds:
                        log_file_2 = os.path.join(log_dir, f'epoch{epoch}_topk{topk}_D{D}_rounds{round}_score.txt')
                        cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} --max-len-a {max_len_a} --max-len-b {max_len_b} > {log_file_2} 2>&1'
                        if 'valid' in dataset:
                            cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} --gen-subset valid --max-size 3000 --seed 1234 --max-len-a {max_len_a} --max-len-b {max_len_b} > {log_file_2} 2>&1'
                        fouts[gpu].write(cmd2+'\n')
                        gpu += 1
                        gpu = gpu % 4
        for i in range(4):
            fouts[i].close()
    else:
        for topk in topks:
            for round in rounds:
                for D in Ds:
                    log_file_1 = os.path.join(log_dir, f'topk{topk}_D{D}_rounds{round}_speed.txt')
                    log_file_2 = os.path.join(log_dir, f'topk{topk}_D{D}_rounds{round}_score.txt')
                   
                    cmd1 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --D {D} > {log_file_1} 2>&1'
                    cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} > {log_file_2} 2>&1'
        
                    if 'times2' in dataset:
                        cmd1 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --D {D} --timesx 2 > {log_file_1} 2>&1'
                        cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} --timesx 2 > {log_file_2} 2>&1'
                        assert 'valid' not in dataset
                    if 'times4' in dataset:
                        cmd1 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --D {D} --timesx 4 > {log_file_1} 2>&1'
                        cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} --timesx 4 > {log_file_2} 2>&1'
                        assert 'valid' not in dataset
        
                    if 'valid' in dataset:
                        cmd1 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --D {D} --gen-subset valid --max-size 3000 --seed 1234 > {log_file_1} 2>&1'
                        cmd2 = f'CUDA_VISIBLE_DEVICES={gpu} fairseq-generate {data_path}  --path {path} --batch-size 1 --topk {topk} --rounds {round} --remove-bpe --cscore -4 --D {D} --gen-subset valid --max-size 3000 --seed 1234 > {log_file_2} 2>&1'
                    fouts[gpu].write(cmd1+'\n')
                    fouts[gpu].write(cmd2+'\n')
                    gpu += 1
                    gpu = gpu % 4
        for i in range(4):
            fouts[i].close()
