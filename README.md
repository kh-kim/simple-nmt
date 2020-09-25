# Simple Neural Machine Translation (Simple-NMT)

This repo contains a simple source code for advanced neural machine translation based on sequence-to-sequence. Most open sources are unnecessarily too complicated, so those have too many features more than people's expected. Therefore, I hope that this repo can be a good solution for people who doesn't want unnecessarily many features.

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpadv/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Features

- [LSTM sequence-to-seuqnce with attention](http://aclweb.org/anthology/D15-1166)
- [Transformer](https://arxiv.org/abs/1706.03762)
- Reinforcement learning for fine-tuning like [Minimum Risk Training (MRT)](https://arxiv.org/abs/1512.02433)
- Beam search with mini-batch in parallel
- [Dual Supervised Learning](https://arxiv.org/abs/1707.00415)

### Implemented Equations

#### Maximum Likelihood Estimation (MLE)

<p align="center"><img src="/tex/287076bf1e54cf080ee5e6c6d9432270.svg?invert_in_darkmode&sanitize=true" align=middle width=194.2640469pt height=129.49760504999998pt/></p>

#### Minimum Risk Training (MRT)

<p align="center"><img src="/tex/e2178b9206e88ee3c33c657344023294.svg?invert_in_darkmode&sanitize=true" align=middle width=552.7717536pt height=74.31972569999999pt/></p>

#### Dual Supervised Learning (DSL)

<p align="center"><img src="/tex/ca879c9675054d376d7486ef21cc5317.svg?invert_in_darkmode&sanitize=true" align=middle width=779.88238845pt height=74.2718922pt/></p>

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
<img src="/tex/488e526633f52bb82913893bfd66202f.svg?invert_in_darkmode&sanitize=true" align=middle width=1433.23214265pt height=1183.5616452pt/> python train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko \
--gpu_id 0 --batch_size 128 --n_epochs 30 --max_length 100 --dropout .2 \
--word_vec_size 512 --hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 2 \
--lr 1e-3 --lr_step 0 --use_adam --rl_n_epochs 0 \
--model_fn ./model.pth
```

#### To continue with RL training

```bash
<img src="/tex/c1ec5a3bd1265ad3b7234b82fcd34908.svg?invert_in_darkmode&sanitize=true" align=middle width=700.2745958999999pt height=118.35616320000003pt/> python train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko \
--gpu_id 0 --batch_size 128 --n_epochs 30 --max_length 100 --dropout .2 \
--hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 32 \
--lr 1e-3 --lr_step 0 --use_adam --use_transformer --rl_n_epochs 0 \
--model_fn ./model.pth
```

#### Dual Supervised Learning

LM Training:
```bash
<img src="/tex/dfd468b72b7d81c4630286ce247fa3af.svg?invert_in_darkmode&sanitize=true" align=middle width=232.9229991pt height=45.84475500000001pt/>
```

### Inference

```bash
<img src="/tex/99822903ed6229addc7fc49a2440b70a.svg?invert_in_darkmode&sanitize=true" align=middle width=1159.56336705pt height=394.5205473pt/> python translate.py --model_fn ./model.pth --gpu_id 0 --lang enko < test.txt > test.result.txt
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
|n_epochs|30|30+40|5+25|
|optimizer|Adam|SGD|Adam|
|lr|1e-3|1e-2|1e-3|
|max_grad_norm|1e+8|5|1e+8|

### Results

Following table shows a evaluation result for each algorithm.

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

### Samples

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
