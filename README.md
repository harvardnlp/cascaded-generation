# Cascade Transformer


# Usage:

```
CUDA_VISIBLE_DEVICES=0 fairseq-generate ../fairseq_vanilla/data-bin/wmt17_en_de_distill --path /n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_70_240000.pt --batch-size 1 --beam 64 --remove-bpe --D 3 --rounds 5 --usenew 0
```

# Multi-GPU version:

```
git checkout distributed
```


```
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-generate ../fairseq_vanilla/data-bin/wmt17_en_de_distill --path /n/rush_lab/users/y/checkpoints/barrier/en-de-distill/checkpoint_70_240000.pt  --batch-size 1 --beam 64 --remove-bpe --D 3 --rounds 5 --ngpus 4
```
