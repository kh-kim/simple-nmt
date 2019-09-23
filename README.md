# Simple Neural Machine Translation (Simple-NMT)

This repo contains a simple source code for advanced neural machine translation based on sequence-to-sequence. Most open sources are unnecessarily too complicated, so those have too many features more than people's expected. Therefore, I hope that this repo can be a good solution for people who doesn't want unnecessarily many features.

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpadv/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Features

- [LSTM sequence-to-seuqnce with attention](http://aclweb.org/anthology/D15-1166)
- [Transformer](https://arxiv.org/abs/1706.03762)
- Reinforcement learning for fine-tuning like [Minimum Risk Training (MRT)](https://arxiv.org/abs/1512.02433)
- Beam search with mini-batch in parallel
- [Dual Supervised Learning](https://arxiv.org/abs/1707.00415)

## Pre-requisite

- Python 3.6 or higher
- PyTorch 1.1 or higher
- TorchText 0.3 or higher

## Usage

### Split corpus to train-set and valid-set

```bash
$ python ./data/build_corpus.py -h
usage: build_corpus.py [-h] --input INPUT --lang LANG --output OUTPUT
                       [--valid_ratio VALID_RATIO] [--test_ratio TEST_RATIO]
                       [--no_shuffle]
```

example usage:

```bash
$ ls ./data
corpus.en  corpus.ko
$ python ./data/build_corpus.py --input ./data/corpus --lang enko --output ./data/corpus
total src lines: 384105
total tgt lines: 384105
write 7682 lines to ./data/corpus.valid.en
write 7682 lines to ./data/corpus.valid.ko
write 376423 lines to ./data/corpus.train.en
write 376423 lines to ./data/corpus.train.ko
```

### Training

```bash
$ python train.py -h
usage: train.py [-h] --model_fn MODEL_FN --train TRAIN --valid VALID --lang
                LANG [--gpu_id GPU_ID] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--verbose VERBOSE]
                [--init_epoch INIT_EPOCH] [--max_length MAX_LENGTH]
                [--dropout DROPOUT] [--word_vec_size WORD_VEC_SIZE]
                [--hidden_size HIDDEN_SIZE] [--n_layers N_LAYERS]
                [--max_grad_norm MAX_GRAD_NORM] [--use_adam] [--lr LR]
                [--lr_step LR_STEP] [--lr_gamma LR_GAMMA]
                [--lr_decay_start LR_DECAY_START] [--use_noam_decay]
                [--lr_n_warmup_steps LR_N_WARMUP_STEPS] [--rl_lr RL_LR]
                [--rl_n_samples RL_N_SAMPLES] [--rl_n_epochs RL_N_EPOCHS]
                [--rl_init_epoch RL_INIT_EPOCH] [--rl_n_gram RL_N_GRAM]
                [--dsl] [--lm_n_epochs LM_N_EPOCHS]
                [--lm_batch_size LM_BATCH_SIZE] [--dsl_n_epochs DSL_N_EPOCHS]
                [--dsl_lambda DSL_LAMBDA] [--use_transformer]
                [--n_splits N_SPLITS]

optional arguments:
  -h, --help            show this help message and exit
  --model_fn MODEL_FN   Model file name to save. Additional information would
                        be annotated to the file name.
  --train TRAIN         Training set file name except the extention. (ex:
                        train.en --> train)
  --valid VALID         Validation set file name except the extention. (ex:
                        valid.en --> valid)
  --lang LANG           Set of extention represents language pair. (ex: en +
                        ko --> enko)
  --gpu_id GPU_ID       GPU ID to train. Currently, GPU parallel is not
                        supported. -1 for CPU. Default=-1
  --batch_size BATCH_SIZE
                        Mini batch size for gradient descent. Default=32
  --n_epochs N_EPOCHS   Number of epochs to train. Default=15
  --verbose VERBOSE     VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE
                        = 0, 1, 2. Default=2
  --init_epoch INIT_EPOCH
                        Set initial epoch number, which can be useful in
                        continue training. Default=1
  --max_length MAX_LENGTH
                        Maximum length of the training sequence. Default=80
  --dropout DROPOUT     Dropout rate. Default=0.2
  --word_vec_size WORD_VEC_SIZE
                        Word embedding vector dimension. Default=512
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM. Default=768
  --n_layers N_LAYERS   Number of layers in LSTM. Default=4
  --max_grad_norm MAX_GRAD_NORM
                        Threshold for gradient clipping. Default=5.0
  --use_adam            Use Adam as optimizer instead of SGD. Other lr
                        arguments should be changed.
  --lr LR               Initial learning rate. Default=1.0
  --lr_step LR_STEP     Number of epochs for each learning rate decay.
                        Default=1
  --lr_gamma LR_GAMMA   Learning rate decay rate. Default=0.5
  --lr_decay_start LR_DECAY_START
                        Learning rate decay start at. Default=10
  --use_noam_decay      Use Noam learning rate decay, which is described in
                        "Attention is All You Need" paper.
  --lr_n_warmup_steps LR_N_WARMUP_STEPS
                        Number of warming up steps for Noam learning rate
                        decay. Default=48000
  --rl_lr RL_LR         Learning rate for reinforcement learning. Default=0.01
  --rl_n_samples RL_N_SAMPLES
                        Number of samples to get baseline. Default=1
  --rl_n_epochs RL_N_EPOCHS
                        Number of epochs for reinforcement learning.
                        Default=10
  --rl_init_epoch RL_INIT_EPOCH
                        Set initial epoch number for RL, which can be useful
                        in continue training. Default=1
  --rl_n_gram RL_N_GRAM
                        Maximum number of tokens to calculate BLEU for
                        reinforcement learning. Default=6
  --dsl                 Training with Dual Supervised Learning method.
  --lm_n_epochs LM_N_EPOCHS
                        Number of epochs for language model training.
                        Default=10
  --lm_batch_size LM_BATCH_SIZE
                        Batch size for language model training. Default=512
  --dsl_n_epochs DSL_N_EPOCHS
                        Number of epochs for Dual Supervised Learning. '--
                        n_epochs' - '--dsl_n_epochs' will be number of epochs
                        for pretraining (without regularization term).
  --dsl_lambda DSL_LAMBDA
                        Lagrangian Multiplier for regularization term.
                        Default=0.001
  --use_transformer     Set model architecture as Transformer.
  --n_splits N_SPLITS   Number of heads in multi-head attention in
                        Transformer. Default=8
```

example usage:

#### Seq2Seq

```bash
$ python train.py --model_fn ./models/enko.pth --train ./data/corpus.train --valid ./data/corpus.valid --lang enko --gpu_id 0  --batch_size 64 --n_epochs 15 --dropout .2 --word_vec_size 512 --hidden_size 768 --n_layers 4 --lr 1. --lr_step 1 --lr_gamma .5 --lr_decay_start 10 --rl_n_epochs 10
```

#### Transformer

Using Noam learning rate decay for training:

```bash
$ python train.py --model_fn ./models/enko.transformer.pth --train ./data/big_corpus.train --valid ./data/big_corpus.valid --lang enko --gpu_id 0 --batch_size 64 --n_epochs 15 --dropout .1 --hidden_size 512 --n_layers 6 --max_grad_norm 1e+10 --use_adam --lr .0044 --use_noam_decay --lr_n_warmup_steps 48000 --use_transformer --n_splits 8
```

In case, using SGD for training -- I prefer this one:

```bash
python ./train.py --model_fn ./models/enko.transformer.sgd.pth --train ./data/big_corpus.train --valid ./data/big_corpus.valid --lang enko --gpu_id 0 --batch_size 80 --n_epochs 20 --dropout .1 --hidden_size 512 --n_layers 6 --max_grad_norm 2. --lr 1. --lr_step 1 --lr_gamma .8 --lr_decay_start 5 --use_transformer --n_splits 8
```

You may need to change the argument parameters.

### Inference

```bash
$ python translate.py -h
usage: translate.py [-h] --model MODEL [--gpu_id GPU_ID]
                    [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH]
                    [--n_best N_BEST] [--beam_size BEAM_SIZE] [--lang LANG]
                    [--length_penalty LENGTH_PENALTY]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model file name to use
  --gpu_id GPU_ID       GPU ID to use. -1 for CPU. Default=-1
  --batch_size BATCH_SIZE
                        Mini batch size for parallel inference. Default=128
  --max_length MAX_LENGTH
                        Maximum sequence length for inference. Default=255
  --n_best N_BEST       Number of best inference result per sample. Default=1
  --beam_size BEAM_SIZE
                        Beam size for beam search. Default=5
  --lang LANG           Source language and target language. Example: enko
  --length_penalty LENGTH_PENALTY
                        Length penalty parameter that higher value produce
                        shorter results. Default=1.2
```

example usage:

```bash
$ python translate.py --model ./model/enko.12.1.18-3.24.1.37-3.92.pth --gpu_id 0 --batch_size 128 --beam_size 8
```

You may also need to change the argument parameters.

## Evaluation

To evaluate the implementation, I trained my own corpus from many website crawling.

|Corpus|Lang1|Lang2|#Lines|Lang1 #Words|Lang2 #Words|
|-|-|-|-|-|-|
|Train-set|en|ko|2,814,676|40,643,480|49,735,827|
|Valid-set|en|ko|9,885|142,822|175,569|

Also, I tested Minimum Risk Training (MRT). After 18 epochs of training with Maximum Likelihood Estimation (MLE), I trained 10 more epochs with MRT. Below table shows that MRT has much better BLEU than MLE.

|||koen|||enko||
|-|-|-|-|-|-|-|
|epoch|train BLEU|valid BLEU|real BLEU|train BLEU|valid BLEU|real BLEU|
|18|||23.56|||28.15|
|19|25.75|29.1943|23.43|24.6|27.7351|29.73
|20|26.19|29.2517|24.13|25.25|27.818|29.22
|21|26.68|29.3997|24.15|25.64|27.8878|28.77
|22|27.12|29.4438|23.89|26.04|27.9814|29.74
|23|27.22|29.4003|24.13|26.16|28.0581|29.03
|24|27.26|29.477|25.09|26.19|28.0924|29.83
|25|27.54|29.5276|25.17|26.27|28.1819|28.9
|26|27.53|29.6685|24.64|26.37|28.171|29.45
|27|27.78|29.618|24.65|26.63|28.241|28.87
|28|27.73|29.7087|24.54|26.83|28.3358|29.11

Below table shows that result from both MLE and MRT in Korean-English translation task.

|INPUT|REF|MLE|MRT|
|-|-|-|-|
|우리는 또한 그 지역의 생선 가공 공장에서 심한 악취를 내며 썩어가는 엄청난 양의 생선도 치웠습니다.|We cleared tons and tons of stinking, rotting fish carcasses from the local fish processing plant.|We also had a huge stink in the fish processing plant in the area, smelling havoc with a huge amount of fish.|We also cleared a huge amount of fish that rot and rot in the fish processing factory in the area.|
|회사를 이전할 이상적인 장소이다.|It is an ideal place to relocate the company.|It's an ideal place to transfer the company.|It's an ideal place to transfer the company.|
|나는 이것들이 내 삶을 바꾸게 하지 않겠어.|I won't let this thing alter my life.|I'm not gonna let these things change my life.|I won't let these things change my life.|
|사람들이 슬퍼보인다.|Their faces appear tearful.|People seem to be sad.|People seem to be sad.|
|아냐, 그런데 넌 그렇다고 생각해.|No, but I think you do.|No, but I think you do.|No, but you think it's.|
|하지만, 나는 나중에 곧 잠들었다.|But I fell asleep shortly afterwards.|However, I fell asleep in a moment.|However, I fell asleep soon afterwards.|
|하지만 1997년 아시아에 외환위기가 불어닥쳤다.|But Asia was hit hard by the 1997 foreign currency crisis.|In 1997, however, the financial crisis in Asia has become a reality for Asia.|But in 1997, the foreign currency crisis was swept in Asia.|
|메이저 리그 공식 웹사이트에 따르면, 12월 22일, 추씨는 텍사스 레인져스와 7년 계약을 맺었다.|According to Major League Baseball's official website, on Dec. 22, Choo signed a seven year contract with the Texas Rangers.|According to the Major League official website on December 22, Choo signed a seven-year contract with Texas Rangers in Texas|According to the Major League official website on December 22, Choo made a seven-year contract with Texas Rangers.|
|한 개인.|a private individual|a person of personal importance|a personal individual|
|도로에 차가 꼬리를 물고 늘어서있다.|The traffic is bumper to bumper on the road.|The road is on the road with a tail.|The road is lined with tail on the road.|
|내가 그렇게 늙지 않았다는 점을 지적해도 될까요.|Let me point out that I'm not that old.|You can point out that I'm not that old.|You can point out that I'm not that old.|
|닐슨 시청률은 15분 단위 증감으로 시청률을 측정하므로, ABC, NBC, CBS 와 Fox 의 순위를 정하지 않았다.|Nielsen had no ratings for ABC, NBC, CBS and Fox because it measures their viewership in 15-minute increments.|The Nielsen ratings measured the viewer's ratings with increments for 15-minute increments, so they did not rank ABC, NBC, CBS and Fox.|Nielson ratings measured ratings with 15-minute increments, so they did not rank ABC, NBC, CBS and Fox.|
|다시말해서, 학교는 교사 부족이다.|In other words, the school is a teacher short.|In other words, school is a teacher short of a teacher.|In other words, school is a lack of teacher.|
|그 다음 몇 주 동안에 사태가 극적으로 전환되었다.|Events took a dramatic turn in the weeks that followed.|The situation has been dramatically changed for the next few weeks.|The situation was dramatically reversed for the next few weeks.|
|젊은이들을 물리학에 대해 흥미를 붙일수 있게 할수 있는 가장 좋은 사람은 졸업생 물리학자이다.|The best possible person to excite young people about physics is a graduate physicist.|The best person to be able to make young people interested in physics is a self-thomac physicist.|The best person to make young people interested in physics is a graduate physicist.|
|5월 20일, 인도는 팔로디 마을에서 충격적인 기온인 섭씨 51도를 달성하며, 가장 더운 날씨를 기록했습니다.|On May 20, India recorded its hottest day ever in the town of Phalodi with a staggering temperature of 51 degrees Celsius.|On May 20, India achieved its hottest temperatures, even 51 degrees Celsius, in the Palrody village, and recorded the hottest weather.|On May 20, India achieved 51 degrees Celsius, a devastating temperature in Paldydy town, and recorded the hottest weather.|
|내말은, 가끔 바나는 그냥 바나나야.|I mean, sometimes a banana is just a banana.|I mean, sometimes a banana is just a banana.|I mean, sometimes a banana is just a banana.|

## Author

|Name|Kim, Ki Hyun|
|-|-|
|email|pointzz.ki@gmail.com|
|github|https://github.com/kh-kim/|
|linkedin|https://www.linkedin.com/in/ki-hyun-kim/|

## References

- [[Luong et al., 2015](http://aclweb.org/anthology/D15-1166)] Effective Approaches to Attention-based Neural Machine Translation
- [[Shen et al., 2015](https://arxiv.org/abs/1512.02433)] Minimum Risk Training for Neural Machine Translation
- [[Sennrich et al., 2016](http://www.aclweb.org/anthology/P16-1162)] Neural Machine Translation of Rare Words with Subword Units
- [[Wu et al, 2016](https://arxiv.org/abs/1609.08144)] Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
- [[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)] Attention is All You Need
- [[Xia et al., 2017](https://arxiv.org/abs/1707.00415)] Dual Supervised Learning
