# Simple Neural Machine Translation Toolkit
This repo contains a simple source code for advanced neural machine translation based on sequence-to-sequence. Most open sources are unnecessarily too complicated, so those have too many features more than people's expected. Therefore, I hope that this repo can be a good solution for people who doesn't want unnecessarily many features.

## Basic Features

1. LSTM sequence-to-seuqnce with attention
2. Beam search with mini-batch in parallel

## Pre-requisite

- Python 3.6 or higher
- PyTorch 0.4 or higher
- TorchText 0.3 or higher

## Usage

### 0. Build Corpus

```
$ python ./data/build_corpus.py -h
usage: build_corpus.py [-h] -input INPUT -lang LANG -output OUTPUT
                       [-valid_ratio VALID_RATIO] [-test_ratio TEST_RATIO]
                       [-no_shuffle]
```

example usage:
```
$ ls ./data
corpus.en  corpus.ko
$ python ./data/build_corpus.py -input ./data/corpus -lang enko -output ./data/corpus
total src lines: 384105
total tgt lines: 384105
write 7682 lines to ./data/corpus.valid.en
write 7682 lines to ./data/corpus.valid.ko
write 376423 lines to ./data/corpus.train.en
write 376423 lines to ./data/corpus.train.ko
```

### 1. Training

```
$ python train.py -h
usage: train.py [-h] -model MODEL -train TRAIN -valid VALID -lang LANG
                [-gpu_id GPU_ID] [-batch_size BATCH_SIZE] [-n_epochs N_EPOCHS]
                [-print_every PRINT_EVERY] [-early_stop EARLY_STOP]
                [-max_length MAX_LENGTH] [-dropout DROPOUT]
                [-word_vec_dim WORD_VEC_DIM] [-hidden_size HIDDEN_SIZE]
                [-n_layers N_LAYERS] [-max_grad_norm MAX_GRAD_NORM] [-adam]
                [-lr LR] [-min_lr MIN_LR]
                [-lr_decay_start_at LR_DECAY_START_AT] [-lr_slow_decay]
```

example usage:
```
$ python train.py -model ./models/enko.small_corpus.pth -train ./data/corpus.train -valid ./data/corpus.valid -lang enko -gpu_id 0 -word_vec_dim 256 -hidden_size 512 -batch_size 32 -n_epochs 18
```

You may need to change the argument parameters.

### 2. Inference

```
$ python translate.py -h
usage: translate.py [-h] -model MODEL [-gpu_id GPU_ID]
                    [-batch_size BATCH_SIZE] [-max_length MAX_LENGTH]
                    [-n_best N_BEST] [-beam_size BEAM_SIZE]
```

example usage:
```
$ python translate.py -model ./model/enko.small_corpus.12.1.18-3.24.1.37-3.92.pth -gpu_id 0 -batch_size 128 -beam_size 5
```

You may also need to change the argument parameters.