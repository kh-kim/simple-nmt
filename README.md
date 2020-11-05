# Simple Neural Machine Translation (Simple-NMT)

This repo contains a simple source code for advanced neural machine translation based on sequence-to-sequence. Most open sources are unnecessarily too complicated, so those have too many features more than people's expected. Therefore, I hope that this repo can be a good solution for people who doesn't want unnecessarily many features.

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpadv/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Features

- [LSTM sequence-to-sequence with attention](http://aclweb.org/anthology/D15-1166)
- [Transformer](https://arxiv.org/abs/1706.03762)
  - Pre-Layer Normalized Transformer
  - Rectified Adam
- Reinforcement learning for fine-tuning like [Minimum Risk Training (MRT)](https://arxiv.org/abs/1512.02433)
- [Dual Supervised Learning](https://arxiv.org/abs/1707.00415)
- Beam search with mini-batch in parallel

### Implemented Optimization Algorithms

#### Maximum Likelihood Estimation (MLE)

<!-- $$\begin{gathered}
\mathcal{D}=\{(x_i,y_i\}_{i=1}^N \\
\\
\hat{\theta}\leftarrow\theta-\eta\nabla_\theta\mathcal{L}(\theta) \\
\mathcal{L}(\theta)=-\sum_{i=1}^N{\log{P(y_i|x_i;\theta)}}
\end{gathered}$$ -->

![](./tex/287076bf1e54cf080ee5e6c6d9432270.svg)

#### Minimum Risk Training (MRT)

<!-- $$\begin{gathered}
\nabla_\theta\mathcal{L}(\theta)=\nabla_\theta\sum_{i=1}^N{}{
    -\Big(\text{reward}(y_i,\hat{y}_i)-\frac{1}{K}\sum_{k=1}^K{
        \text{reward}(y_i,\hat{y}_{i,k})
    }\Big)\times\log{P(\hat{y}_i|x_i;\theta)}
}, \\
\text{where }\hat{y}_i\sim{P(\text{y}|x_i;\theta)}.
\end{gathered}$$ -->

![](./tex/9adfbd5b3e9441850e393098e3337a1a.svg)

#### Dual Supervised Learning (DSL)

<!-- $$\begin{gathered}
\theta_{x\rightarrow{y}}\leftarrow\theta_{x\rightarrow{y}}-\eta\nabla_{\theta_{x\rightarrow{y}}}\mathcal{L}(\theta_{x\rightarrow{y}}) \\
\mathcal{L}(\theta_{x\rightarrow{y}})=-\sum_{i=1}^N{
    \log{P(y_i|x_i;\theta_{x\rightarrow{y}})}
}+\bigg\|
    \Big(\log{P(y_i|x_i;\theta_{x\rightarrow{y}})}+\log{\hat{P}(x_i)}\Big)
    -\Big(\log{P(x_i|y_i;\theta_{y\rightarrow{x}})}+\log{\hat{P}(y_i)}\Big)
\bigg\|_2^2
\end{gathered}$$ -->

![](./tex/ca879c9675054d376d7486ef21cc5317.svg)

## Pre-requisite

- Python 3.6 or higher
- PyTorch 1.6 or higher
- TorchText 0.5 or higher
- PyTorch Ignite
- [torch-optimizer 0.0.1a15](https://pypi.org/project/torch-optimizer/)

## Usage

I recommend to use corpora from [AI-Hub](http://www.aihub.or.kr/), if you are trying to build Kor/Eng machine translation.

### Training

```bash
>> python train.py -h
usage: train.py [-h] --model_fn MODEL_FN --train TRAIN --valid VALID --lang
                LANG [--gpu_id GPU_ID] [--off_autocast]
                [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                [--verbose VERBOSE] [--init_epoch INIT_EPOCH]
                [--max_length MAX_LENGTH] [--dropout DROPOUT]
                [--word_vec_size WORD_VEC_SIZE] [--hidden_size HIDDEN_SIZE]
                [--n_layers N_LAYERS] [--max_grad_norm MAX_GRAD_NORM]
                [--iteration_per_update ITERATION_PER_UPDATE] [--lr LR]
                [--lr_step LR_STEP] [--lr_gamma LR_GAMMA]
                [--lr_decay_start LR_DECAY_START] [--use_adam] [--use_radam]
                [--rl_lr RL_LR] [--rl_n_samples RL_N_SAMPLES]
                [--rl_n_epochs RL_N_EPOCHS] [--rl_n_gram RL_N_GRAM]
                [--rl_reward RL_REWARD] [--use_transformer]
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
  --off_autocast        Turn-off Automatic Mixed Precision (AMP), which speed-
                        up training.
  --batch_size BATCH_SIZE
                        Mini batch size for gradient descent. Default=32
  --n_epochs N_EPOCHS   Number of epochs to train. Default=20
  --verbose VERBOSE     VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE
                        = 0, 1, 2. Default=2
  --init_epoch INIT_EPOCH
                        Set initial epoch number, which can be useful in
                        continue training. Default=1
  --max_length MAX_LENGTH
                        Maximum length of the training sequence. Default=100
  --dropout DROPOUT     Dropout rate. Default=0.2
  --word_vec_size WORD_VEC_SIZE
                        Word embedding vector dimension. Default=512
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM. Default=768
  --n_layers N_LAYERS   Number of layers in LSTM. Default=4
  --max_grad_norm MAX_GRAD_NORM
                        Threshold for gradient clipping. Default=5.0
  --iteration_per_update ITERATION_PER_UPDATE
                        Number of feed-forward iterations for one parameter
                        update. Default=1
  --lr LR               Initial learning rate. Default=1.0
  --lr_step LR_STEP     Number of epochs for each learning rate decay.
                        Default=1
  --lr_gamma LR_GAMMA   Learning rate decay rate. Default=0.5
  --lr_decay_start LR_DECAY_START
                        Learning rate decay start at. Default=10
  --use_adam            Use Adam as optimizer instead of SGD. Other lr
                        arguments should be changed.
  --use_radam           Use rectified Adam as optimizer. Other lr arguments
                        should be changed.
  --rl_lr RL_LR         Learning rate for reinforcement learning. Default=0.01
  --rl_n_samples RL_N_SAMPLES
                        Number of samples to get baseline. Default=1
  --rl_n_epochs RL_N_EPOCHS
                        Number of epochs for reinforcement learning.
                        Default=10
  --rl_n_gram RL_N_GRAM
                        Maximum number of tokens to calculate BLEU for
                        reinforcement learning. Default=6
  --rl_reward RL_REWARD
                        Method name to use as reward function for RL training.
                        Default=gleu
  --use_transformer     Set model architecture as Transformer.
  --n_splits N_SPLITS   Number of heads in multi-head attention in
                        Transformer. Default=8
```

example usage:

#### Seq2Seq

```bash
>> python train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko \
--gpu_id 0 --batch_size 128 --n_epochs 30 --max_length 100 --dropout .2 \
--word_vec_size 512 --hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 2 \
--lr 1e-3 --lr_step 0 --use_adam --rl_n_epochs 0 \
--model_fn ./model.pth
```

#### To continue with RL training

```bash
>> python continue_train.py --load_fn ./model.pth --model_fn ./model.rl.pth \
--init_epoch 31 --iteration_per_update 1 --max_grad_norm 5
```

#### Transformer

```bash
>> python train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko \
--gpu_id 0 --batch_size 128 --n_epochs 30 --max_length 100 --dropout .2 \
--hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 32 \
--lr 1e-3 --lr_step 0 --use_adam --use_transformer --rl_n_epochs 0 \
--model_fn ./model.pth
```

#### Dual Supervised Learning

LM Training:
```bash
>> python lm_train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko \
--gpu_id 0 --batch_size 256 --n_epochs 20 --max_length 64 --dropout .2 \
--word_vec_size 512 --hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 \
--model_fn ./lm.pth
```

DSL using pretrained LM:
```bash
>> python dual_train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko \
--gpu_id 0 --batch_size 64 --n_epochs 40 --max_length 64 --dropout .2 \
--word_vec_size 512 --hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 4 \
--dsl_n_warmup_epochs 30 --dsl_lambda 1e-2 \
--lm_fn ./lm.pth \
--model_fn ./model.pth
```

Note that I recommend to use different 'max_grad_norm value' (e.g. 5) for after warm-up training. You can use 'continue_dual_train.py' to change 'max_grad_norm' argument.

### Inference

You can translate any sentence via standard input and output.

```bash
>> python translate.py -h
usage: translate.py [-h] --model_fn MODEL_FN [--gpu_id GPU_ID]
                    [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH]
                    [--n_best N_BEST] [--beam_size BEAM_SIZE] [--lang LANG]
                    [--length_penalty LENGTH_PENALTY]

optional arguments:
  -h, --help            show this help message and exit
  --model_fn MODEL_FN   Model file name to use
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
>> python translate.py --model_fn ./model.pth --gpu_id 0 --lang enko < test.txt > test.result.txt
```

You may also need to change the argument parameters.

## Evaluation

### Setup

In order to evaluate this project, I used public dataset from [AI-HUB](https://aihub.or.kr/), which provides 1,600,000 pairs of sentence.
I randomly split this data into train/valid/test set by following number of lines each.
In fact, original test set, which has about 200000 lines, is too big to take bunch of evaluations, I reduced it to 1,000 lines.
(In other words, you can get better model, if you put removed 200,000 lines into training set.)

|set|lang|#lines|#tokens|#characters|
|-|-|-|-|-|
|train|en|1,200,000|43,700,390|367,477,362|
||ko|1,200,000|39,066,127|344,881,403|
|valid|en|200,000|7,286,230|61,262,147|
||ko|200,000|6,516,442|57,518,240|
|valid-1000|en|1,000|36,307|305,369|
||ko|1,000|32,282|285,911|
|test-1000|en|1,000|35,686|298,993|
||ko|1,000|31,720|280,126|

Each dataset is tokenized with Mecab/MosesTokenizer and BPE.
After preprocessing, each language has vocabulary size like as below:

|en|ko|
|-|-|
|20,525|29,411|

Also, we have following hyper-parameters for each model to proceed a evaluation.

|parameter|seq2seq|transformer|
|-|-|-|
|batch_size|320|4096|
|word_vec_size|512| - |
|hidden_size|768|768|
|n_layers|4|4|
|n_splits| - |8|
|n_epochs|30|30|

Below is a table for hyper-parameters for each algorithm.

|parameter|MLE|MRT|DSL|
|-|-|-|-|
|n_epochs|30|30 + 40|30 + 10|
|optimizer|Adam|SGD|Adam|
|lr|1e-3|1e-2|1e-2|
|max_grad_norm|1e+8|5|1e+8 $\rightarrow$ 5|

Please, note that MRT has different optimization setup.

### Results

Following table shows a evaluation result for each algorithm.

||enko|koen|
|:-:|:-:|:-:|
|Sequence-to-Sequence|32.53|29.67|
|Sequence-to-Sequence (MRT)|34.04|31.24|
|Sequence-to-Sequence (DSL)|33.47|31.00|
|Transformer|34.96|31.84|
|Transformer (MRT)|-|-|
|Transformer (DSL)|35.48|32.80|

As you can see, Transformer outperforms in ENKO/KOEN task.
I couldn't run MRT on Transformer, due to lack of memory.

Following table shows the result based on beam-size on Sequence-to-Sequence model.
Table shows that beam search improve BLEU score without data adding and model change.

|beam_size|enko|koen|
|:-:|:-:|:-:|
|1|31.65|28.93|
|5|32.53|29.67|
|10|32.48|29.37|

### Samples

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

## References

- [[Luong et al., 2015](http://aclweb.org/anthology/D15-1166)] Effective Approaches to Attention-based Neural Machine Translation
- [[Shen et al., 2015](https://arxiv.org/abs/1512.02433)] Minimum Risk Training for Neural Machine Translation
- [[Sennrich et al., 2016](http://www.aclweb.org/anthology/P16-1162)] Neural Machine Translation of Rare Words with Subword Units
- [[Wu et al, 2016](https://arxiv.org/abs/1609.08144)] Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
- [[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)] Attention is All You Need
- [[Xia et al., 2017](https://arxiv.org/abs/1707.00415)] Dual Supervised Learning
