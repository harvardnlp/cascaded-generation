# Cascade Transformer

Here we provide code to reproduce our results. We provide all training data and training scripts, as well as all pretrained models used in our paper. Our code is built on top of [fairseq](https://github.com/pytorch/fairseq) and [pytorch-strcut](https://github.com/harvardnlp/pytorch-struct).

## Prerequisites

```
pip install -qU git+https://github.com/harvardnlp/pytorch-struct
pip install -qU git+https://github.com/harvardnlp/genbmm
pip install -q matplotlib
pip install --editable .
```

## Datasets & Pretrained Models

We only include `IWSLT14 De-En` in this repository. The entire folder can be found at [here](https://drive.google.com/drive/folders/1G5Vl150cPyc5EWxxqRdngwUifccQeccN?usp=sharing). Data and model for individual datasets can be found at links below.

* WMT14 En-De Distilled: [data]() [model]()
* WMT14 De-En Distilled: [data](https://drive.google.com/file/d/1jkLf_6VZnG358mf2y6e4RTDi54WkChDI/view?usp=sharing) [model]()
* WMT16 En-Ro Distilled: [data]() [model]()
* WMT16 Ro-En Distilled: [data]() [model]()
* IWSLT14 De-En Distilled: [data](https://drive.google.com/file/d/1F51UMYW-nHx8nhkX3JR1QVygfQBSoS6F/view?usp=sharing) [model]()
* WMT14 En-De: [data]() [model]()
* WMT14 De-En: [data](https://drive.google.com/file/d/1bSOAPb0xw-zgSaIvsOWzVqOJKzxeG9vz/view?usp=sharing) (same as WMT14 En-De) [model]()
* WMT16 En-Ro: [data]() [model]()
* WMT16 Ro-En: [data]() (same as WMT16 En-Ro) [model]()
* IWSLT14 De-En: [data](https://drive.google.com/file/d/1v7Z-23-U5WV8KhlzrepMVR0J69zH-k0R/view?usp=sharing) [model]()

## Usage

### Data Preprocessing

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

### Training

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
    --eval-bleu --eval-bleu-args '{"max_len_a": '$MAX_LEN_A', "max_len_b": '$MAX_LEN_B'}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-dir $SAVE_DIR \
    --max-update $MAX_UPDATES --validation-max-size 3000 \
    --validation-topk 16 --validation-D 3 --validation-rounds 5 --seed 1234 --ngrams 5
```

We use a single GPU to train on IWSLT14 De-En. After training is done, we can use `checkpoints/iwslt14-de-en/checkpoint_best.pt` for generation.


### Generation

As an example, let's generate from the above trained model. We use `topk = 32` and `rounds = 5`.

```
DATASET=iwslt14-de-en
DATA_BIN=data-bin/$DATASET
SAVE_DIR=checkpoints/$DATASET
BATCH_SIZE=1
TOPK=32
rounds=5
MAX_LEN_A=0.941281036889224
MAX_LEN_B=0.8804326732522796
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_BIN --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size $BATCH_SIZE --topk $TOPK --remove-bpe --D 3 --rounds $rounds \
    --max-len-a $MAX_LEN_A --max-len-b $MAX_LEN_B
```

Note that using a model trained on a different dataset requires re-estimating `max-len-a` and `max-len-b`.

### Multi-GPU Generation:

Our approach is amenable to multi-GPU parallelization: we can even get further speedup at batch size 1 using multiple GPUs.

```
NGPUS=3
DATASET=iwslt14-de-en
DATA_BIN=data-bin/$DATASET
SAVE_DIR=checkpoints/$DATASET
TOPK=32
rounds=5
MAX_LEN_A=0.941281036889224
MAX_LEN_B=0.8804326732522796
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-generate $DATA_BIN --path $SAVE_DIR/checkpoint_best.pt \
    --batch-size 1 --topk $TOPK --remove-bpe --D 3 --rounds $rounds --ngpus $NGPUS \
    --max-len-a $MAX_LEN_A --max-len-b $MAX_LEN_B
```


### Training on Other Datasets

#### WMT14 (raw/distilled) En-De/De-En

For preprocessing, we need to use joined dictionary.

```
DATASET=? # dataset dependent
SOURCE_LANG=? # dataset dependent
TARGET_LANG=? # dataset dependent
TEXT=data/$DATASET
DATA_BIN=data-bin/$DATASET
fairseq-preprocess \
    --source-lang $SOURCE_LANG --target-lang $TARGET_LANG \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATA_BIN --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20 --joined-dictionary
```

We train on 3 GPUs.

```
DATASET=? # dataset dependent
MAX_LEN_A=? # dataset dependent
MAX_LEN_B=? # dataset dependent
DATA_BIN=data-bin/$DATASET
SAVE_DIR=checkpoints/$DATASET
ARCH=transformer_wmt_en_de
DROPOUT=0.1
MAX_TOKENS=4096
LR=7e-4
WARMUP_UPDATES=4000
MAX_UPDATES=240000
WEIGHT_DECAY=0.0
CUDA_VISIBLE_DEVICES=0,1,2 python train.py $DATA_BIN --arch $ARCH --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
    --warmup-updates 4000 --lr $LR --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay $WEIGHT_DECAY --max-tokens $MAX_TOKENS --save-dir $SAVE_DIR --update-freq 3 \
    --no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 --dropout $DROPOUT\
    --fp16 --ddp-backend=no_c10d --eval-bleu --eval-bleu-args '{"max_len_a": '$MAX_LEN_A', "max_len_b": '$MAX_LEN_B'}'\
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update $MAX_UPDATES  --validation-max-size 3000 --validation-topk 16 --validation-D 3 --validation-rounds 5 --seed 1234
```

#### WMT16 (raw/distilled) En-Ro/Ro-En

For preprocessing, we need to use joined dictionary.

```
DATASET=? # dataset dependent
SOURCE_LANG=? # dataset dependent
TARGET_LANG=? # dataset dependent
TEXT=data/$DATASET
DATA_BIN=data-bin/$DATASET
fairseq-preprocess \
    --source-lang $SOURCE_LANG --target-lang $TARGET_LANG \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATA_BIN --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20 --joined-dictionary
```

We train on 3 GPUs.

```
DATASET=? # dataset dependent
MAX_LEN_A=? # dataset dependent
MAX_LEN_B=? # dataset dependent
DATA_BIN=data-bin/$DATASET
SAVE_DIR=checkpoints/$DATASET
ARCH=transformer_wmt_en_de
DROPOUT=0.3
MAX_TOKENS=5461
LR=7e-4
WARMUP_UPDATES=10000
MAX_UPDATES=120000
WEIGHT_DECAY=0.01
CUDA_VISIBLE_DEVICES=0,1,2 python train.py $DATA_BIN --arch $ARCH --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
    --warmup-updates $WARMUP_UPDATES --lr $LR --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --weight-decay $WEIGHT_DECAY --max-tokens $MAX_TOKENS --save-dir $SAVE_DIR --update-freq 1 \
    --no-progress-bar --log-format json --log-interval 50 --save-interval-updates 1000 --dropout $DROPOUT\
    --fp16 --ddp-backend=no_c10d --eval-bleu --eval-bleu-args '{"max_len_a": '$MAX_LEN_A', "max_len_b": '$MAX_LEN_B'}'\
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-update $MAX_UPDATES --validation-max-size 3000 --validation-topk 16 --validation-D 3 --validation-rounds 5 --seed 1234
```
