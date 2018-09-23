# Simple Neural Machine Translation (Simple-NMT)

This repo contains a simple source code for advanced neural machine translation based on sequence-to-sequence. Most open sources are unnecessarily too complicated, so those have too many features more than people's expected. Therefore, I hope that this repo can be a good solution for people who doesn't want unnecessarily many features.

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpadv/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Features

- [LSTM sequence-to-seuqnce with attention](http://aclweb.org/anthology/D15-1166)
- Reinforcement learning for fine-tuning like [Minimum Risk Training (MRT)](https://arxiv.org/abs/1512.02433)
- Beam search with mini-batch in parallel

## Pre-requisite

- Python 3.6 or higher
- PyTorch 0.4 or higher
- TorchText 0.3 or higher (You may need to install from [github](https://github.com/pytorch/text).)

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
usage: train.py [-h] --model MODEL --train TRAIN --valid VALID --lang LANG
                [--gpu_id GPU_ID] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--print_every PRINT_EVERY]
                [--early_stop EARLY_STOP] [--max_length MAX_LENGTH]
                [--dropout DROPOUT] [--word_vec_dim WORD_VEC_DIM]
                [--hidden_size HIDDEN_SIZE] [--n_layers N_LAYERS]
                [--max_grad_norm MAX_GRAD_NORM] [--adam] [--lr LR]
                [--min_lr MIN_LR] [--lr_decay_start_at LR_DECAY_START_AT]
                [--lr_slow_decay] [--lr_decay_rate LR_DECAY_RATE]
                [--rl_lr RL_LR] [--n_samples N_SAMPLES]
                [--rl_n_epochs RL_N_EPOCHS] [--rl_n_gram RL_N_GRAM]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model file name to save. Additional information would
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
  --n_epochs N_EPOCHS   Number of epochs to train. Default=18
  --print_every PRINT_EVERY
                        Number of gradient descent steps to skip printing the
                        training status. Default=1000
  --early_stop EARLY_STOP
                        The training will be stopped if there is no
                        improvement this number of epochs. Default=-1
  --max_length MAX_LENGTH
                        Maximum length of the training sequence. Default=80
  --dropout DROPOUT     Dropout rate. Default=0.2
  --word_vec_dim WORD_VEC_DIM
                        Word embedding vector dimension. Default=512
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM. Default=768
  --n_layers N_LAYERS   Number of layers in LSTM. Default=4
  --max_grad_norm MAX_GRAD_NORM
                        Threshold for gradient clipping. Default=5.0
  --adam                Use Adam instead of using SGD.
  --lr LR               Initial learning rate. Default=1.0
  --min_lr MIN_LR       Minimum learning rate. Default=.000001
  --lr_decay_start_at LR_DECAY_START_AT
                        Start learning rate decay from this epoch.
  --lr_slow_decay       Decay learning rate only if there is no improvement on
                        last epoch.
  --lr_decay_rate LR_DECAY_RATE
                        Learning rate decay rate. Default=0.5
  --rl_lr RL_LR         Learning rate for reinforcement learning. Default=.01
  --n_samples N_SAMPLES
                        Number of samples to get baseline. Default=1
  --rl_n_epochs RL_N_EPOCHS
                        Number of epochs for reinforcement learning.
                        Default=10
  --rl_n_gram RL_N_GRAM
                        Maximum number of tokens to calculate BLEU for
                        reinforcement learning. Default=6
```

example usage:

```bash
$ python train.py --model ./models/enko.pth --train ./data/corpus.train --valid ./data/corpus.valid --lang enko --gpu_id 0 --word_vec_dim 256 --hidden_size 512 --batch_size 32 --n_epochs 15 --rl_n_epochs 10 --early_stop -1
```

You may need to change the argument parameters.

### Inference

```bash
$ python translate.py -h
usage: translate.py [-h] --model MODEL [--gpu_id GPU_ID]
                    [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH]
                    [--n_best N_BEST] [--beam_size BEAM_SIZE]

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
```

example usage:

```bash
$ python translate.py --model ./model/enko.12.1.18-3.24.1.37-3.92.pth --gpu_id 0 --batch_size 128 --beam_size 5
```

You may also need to change the argument parameters.

## Evaluation

|Corpus|Lang1|Lang2|#Lines|Lang1 #Words|Lang2 #Words|
|-|-|-|-|-|-|
|Train-set|en|ko|2,814,676|40,643,480|49,735,827|
|Valid-set|en|ko|9,885|142,822|175,569|

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

|REF|MLE|MRT|
|-|-|-|
|사탄. 톰 레일리 : 가사 전부를 듣기는 좀 어렵네요, 그래서 저는-- (웃음) 그래서 저는 여러분을 조금이라도 도우려 합니다.|톰: 이제 메세지를 듣기가 조금 어렵습니다. (웃음) 그래서 저는 당신을 조금 돕고 싶었습니다.|톰: 이제 메세지를 듣기가 조금 어렵습니다. (웃음) 그래서 저는 여러분을 조금 돕고 싶었습니다. 그래서 저는 여러분을 조금 돕고 싶었습니다.|
|몇 가지 다른 종류의 살모넬라 박테리아가 있고 그것들은 모두 여러분을 아프게 할 수 있습니다.|다양한 종류의 살모넬라 박테리아가 있고, 그들은 여러분을 아프게 할 수 있습니다.|몇 가지 다른 종류의 살모넬라 박테리아가 있고, 모두 여러분을 아프게 할 수 있습니다.|
그녀는 몸을 과학에 주고 떠났다.|그녀는 과학에 몸을 맡기고 있다.|그녀는 과학에 몸을 맡기고 떠난다.|
|마마는 1999년에 한국에서 시작되었습니다.|그 안전국은 1999년에 한국에서 시작했다.|MAMA는 1999년에 한국에서 시작되었다.|
|그들에게는 평범하게 사는 것이 어려울 수 있습니다.|평범한 삶을 살 수 있는 것은 그들에게 힘들 수 있다|평범한 삶을 사는 것은 그들에게는 어려울 수 있다.|
|그의 팔에 입은 총상을 붕대로 감다.|팔에 총상을 입은 상처를 입다.|팔에 총알을 붕대를 바르다.|
|요가는 여러분이 심호흡을 하도록 돕고, 여러분이 폐를 더욱 기능적으로 사용하면 그것은 천식으로부터 여러분을 보호해 줍니다.|요가는 여러분이 깊게 호흡하는데 도움을 주고 폐의 기능적인 사용을 더 많이 할 때 천식을 예방하는데 도움을 줍니다.|요가는 여러분이 깊게 숨쉬는 데 도움이 되고 폐를 더 기능적이게 할 때 천식에 대항하는 것을 돕습니다.|
|미국 유명 연예인 패리스 힐튼의 여동생은 패션 디자이너로서의 직업에 초점을 둘 것이라고 발표했습니다.|미국 연예인 패리스 힐튼의 여동생은 그녀가 패션 디자이너로서의 경력에 집중할 것이라고 발표했다.|미국 유명 연예인 패리스 힐튼 힐튼은 그녀가 패션 디자이너로 그녀의 경력에 집중할 것이라고 발표했다.|
|나는 우리가 가능성이 없다는 것이 두려워.|나는 우리가 가망이 없는 것 같다.|유감스럽게도 우리가 가망이 없는 것 같다.|

## Author

|Name|Kim, Ki Hyun|
|-|-|
|email|pointzz.ki@gmail.com|
|github|https://github.com/kh-kim/|
|linkedin|https://www.linkedin.com/in/ki-hyun-kim/|

## References

- [[Luong et al.2015](http://aclweb.org/anthology/D15-1166)] Effective Approaches to Attention-based Neural Machine Translation
- [[Shen et al.2015](https://arxiv.org/abs/1512.02433)] 
Minimum Risk Training for Neural Machine Translation
- [[Sennrich et al.2016](http://www.aclweb.org/anthology/P16-1162)] Neural Machine Translation of Rare Words with Subword Units