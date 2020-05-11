# Cascade Transformer

# Prerequisites

```
pip install --editable .
```

# Usage:

## Data Preprocessing

Throughout this Readme, we use `IWSLT14 De-En` as an example to show how to reproduce our results. First, we need to figure out the mapping from source length to target length. We simply use linear regression here: `target_length = max-len-a * source_length + max-len-b`, and we need to estimate `max-len-a` and `max-len-b` from training data. Note that directly using `max-len-a=1` and `max-len-b=0` would still reach reasonable performance.

```
python scripts/get_max_len_ab.py data/iwslt14-de-en/train.de data/iwslt14-de-en/train.en
```

Then we would figure out `max-len-a = 0.941281036889224` and `max-len-b = 0.8804326732522796`.

Before training our model, we need to preprocess the training data using `fairseq-preprocess`.

```
DATASET=iwslt14-de-en
TEXT=data/$DATASET
DATA_BIN=data-bin/$DATASET
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATA_BIN \
    --workers 20
```

## Training

```
DATASET=iwslt14-de-en
DATA_BIN=data-bin/$DATASET
SAVE_DIR=checkpoints/$DATASET
ARCH=transformer_iwslt_de_en
DROPOUT=0.3
MAX_TOKENS=4096
LR=5e-4
WARMUP_UPDATES=4000
MAX_UPDATES=120000
WEIGHT_DECAY=0.0001
MAX_LEN_A=0.941281036889224
MAX_LEN_B=0.8804326732522796
CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_BIN --arch $ARCH --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr $LR --lr-scheduler inverse_sqrt \
    --warmup-updates $WARMUP_UPDATES --dropout $DROPOUT --weight-decay $WEIGHT_DECAY \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens $MAX_TOKENS \
    --eval-bleu --eval-bleu-args '\{\"max_len_a\": $MAX_LEN_A, \"max_len_b\": $MAX_LEN_B\}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-dir $SAVE_DIR \
    --max-update $MAX_UPDATES --validation-max-size 3000 \
    --validation-topk 16 --validation-D 3 --validation-rounds 5 --seed 1234 --ngrams 5
```

We use a single GPU to train on IWSLT14 De-En. After training is done, we can use `checkpoints/iwslt14-de-en/checkpoint_best.pt` for generation.


## Generation


```
CUDA_VISIBLE_DEVICES=0 fairseq-generate ../fairseq_vanilla/data-bin/wmt17_en_de_distill --path /n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_70_240000.pt --batch-size 1 --topk 64 --remove-bpe --D 3 --rounds 5
```

## Multi-GPU Generation:


```
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-generate ../fairseq_vanilla/data-bin/wmt17_en_de_distill --path /n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_70_240000.pt  --batch-size 1 --topk 64 --remove-bpe --D 3 --rounds 5 --ngpus 4
```
