## ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers (Interspeech 2020)

[ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers](https://arxiv.org/abs/2004.09367)

[Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<sup>1*</sup>, [Kihyun Nam](https://github.com/DevKiHyun)<sup>1,2*</sup>, [Jingu Kang](https://github.com/kibitzing)<sup>1</sup>, [Sang-Woo Lee](https://scholar.google.co.kr/citations?user=TMTTMuQAAAAJ)<sup>1</sup>, [Sohee Yang](https://github.com/soheeyang)<sup>1</sup>, [Hyunhoon Jung](https://www.linkedin.com/in/hyunhoon-jung-00a958a1/)<sup>1</sup>, Eunmi Kim<sup>1</sup>, <p>
Hyeji Kim<sup>1</sup>, Soojin Kim<sup>1</sup>, Hyun Ah Kim<sup>1</sup>, [Kyoungtae Doh](https://github.com/ehrudxo)<sup>1</sup>, Chan Kyu Lee<sup>1</sup>, [Nako Sung](https://github.com/nakosung)<sup>1</sup>, Sunghun Kim<sup>1,3</sup> 

<sup>1</sup>Clova AI, NAVER Corp.
<sup>2</sup>Hankuk University on Foreign Studies <p>
<sup>3</sup>The Hong Kong University of Science and Technology <p>
<sup>*</sup> Both authors equally contributed to this work.

Automatic speech recognition (ASR) via call is essential for various applications including AI for contact center (CCAI) services. Despite the advancement of ASR, however, most call speech corpora publicly available were old-fashioned such as Swichboard. Also, most call corpora are in English and mainly focus on open domain scenarios such as audio book. Here we introduce a new large-scale Korean call-based speech corpus under a goal-oriented dialog scenario from more than 11,000 people, i.e. Clova Call corpus (ClovaCall). The raw dataset of ClovaCall includes approximately 112,000 pairs of a short sentence and its corresponding spoken utterance in a restaurant reservation domain. We validate the effectiveness of our dataset with intensive experiments on two state-of-the art ASR models.



## Table of contents 

* [1. Naver ClovaCall dataset contribution](#1-naver-clovacall-dataset-contribution)
    + [The dataset statistics](#the-dataset-statistics)
    + [The dataset structure](#the-dataset-structure)
* [2. Dataset downloading and license](#2-dataset-downloading-and-license)
    + [ClovaCall](#clovacall)
    + [Licenses](#licenses)
* [3. Model](#3-model)
* [4. Dependency](#4-dependency)
* [5. Training and Evaluation](#5-training-and-evaluation)
    + [Run train](#run-train)
    + [Run eval](#run-eval)
* [6. Code license](#6-code-license)
* [7. Reference](#7-reference)
* [8. How to cite](#8-how-to-cite)



## 1. Naver ClovaCall dataset contribution
Call-based customer services are still prevalent in most online and offline industries. 
Our ClovaCall can contribute to ASR models for diverse call-based reservation services, considering that many reservation services share common expression such as working time, location, availability, etc.

ClovaCall has two version, `raw` version and `clean` version. We used `librosa` with the threshold as 25db for silence elimination. The silence-free data is called `clean` version.
### The dataset statistics

Dataset         | Number       | Hour (raw / clean)        |
----------------|--------------|---------------------------|
Raw             | 81,222       | 125 / 67                  |
Train           | 59,662       | 80 / 50                  |
Test            | 1,084        | 1.66 / 0.88                |

### The dataset structure

We provide the json file for ClovaCall with the following structure:
```
ClovaCall.json

[
  {
    "wav" : "42_0603_748_0_03319_00.wav",
    "text" : "단체 할인이 가능한 시간대가 따로 있나요?",
    "speaker_id" : "03319"
  },
  ...,
  {
    "wav" : "42_0610_778_0_03607_01.wav",
    "text" : "애기들이 놀만한 놀이방이 따로 있나요?",
    "speaker_id" : "03607"
  }
]  
```



## 2. Dataset downloading and license
```
To all the materials including speech data distributed here(hereinafter, “MATERIALS”), the following license(hereinafter, “LICENSE”) shall apply. If there is any conflict between the LICENSE and the clovaai/speech_hackathon_19 License(Apache Lincese 2.0) listed on Github, the LICENSE below shall prevail.

1. You are allowed to use the MATERIALS ONLY FOR NON-COMMERCIAL AI(Artificial Intelligence) RESEARCH AND DEVELOPMENT PURPOSES – ANY KIND OF COMMERCIAL USE IS STRICTLY PROHIBITED.
2. You should USE THE MATERIALS AS THEY WERE PROVIDED – ANY KIND OF MODIFICATION, EDITING AND REPRODUCTION TO DATA IS STRICTLY PROHIBITED.
3. You should use the MATERIALS only by/for yourself. You are NOT ALLOWED TO COPY, DISTRIBUTE, PROVIDE, TRANSPORT THE MATERIALS TO ANY 3RD PARTY OR TO THE PUBLIC including uploading the MATERIALS to internet.
4. You should clearly notify the source of the MATERIALS as “NAVER Corp.” when your use the MATERIALS.
5. NAVER Corp. DOES NOT GUARANTEE THE ACCURACY, COMPLETENESS, INTEGRITY, QUALITY OR ADEQUACY OF THE MATERIALS, THUS ARE NOT LIABLE OR RESPONSIBLE FOR THE MATERIALS PROVIDED HERE.

※ Please be noted that since the MATERIALS should be used within the confines of the voice right owner’s agreement (which was reflected in the LICENSE above), your non-compliance of the LICENSE (for example, using the MATERIAL for commercial use or modifying or distributing the MATERIAL) shall also constitute infringement on the voice right owner’s rights, thus may cause expose you to legal claims from the voice right owner.
```

`ClovaCall` dataset can be download for researchers involved in Acdaemic Organizations by applying via [here](https://docs.google.com/forms/d/e/1FAIpQLSf5bm7FtWYeZf8C02mlyZCg32yMrA9_DgKU17oD0migPkEXog/viewform)

`AIhub` dataset can be download from [here](http://www.aihub.or.kr/aidata/105)
(`AIhub` : this is a large-scale Korean open domain dialog corpus from NIA AIHub5, an open data hub site of Korean Govern-ment.) 



## 3. Model

We use two standard ASR models such as Deepspeech2 and LAS for verifying the effectiveness of our proposed ClovaCall. Also, we release baseline source code about LAS.

```
Seq2seq(
  (encoder): EncoderRNN(
    (input_dropout): Dropout(p=0.3, inplace=False)
    (conv): MaskConv(
      (seq_module): Sequential(
        (0): Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Hardtanh(min_val=0, max_val=20, inplace=True)
        (3): Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5))
        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Hardtanh(min_val=0, max_val=20, inplace=True)
      )
    )
    (rnn): LSTM(1312, 512, num_layers=3, dropout=0.3, bidirectional=True)
  )
  (decoder): DecoderRNN(
    (input_dropout): Dropout(p=0.3, inplace=False)
    (rnn): LSTM(1536, 512, num_layers=2, batch_first=True, dropout=0.3)
    (embedding): Embedding(2003, 512)
    (attention): Attention(
      (conv): Conv1d(1, 512, kernel_size=(3,), stride=(1,), padding=(1,))
      (W): Linear(in_features=512, out_features=512, bias=False)
      (V): Linear(in_features=1024, out_features=512, bias=False)
      (fc): Linear(in_features=512, out_features=1, bias=True)
      (tanh): Tanh()
      (softmax): Softmax(dim=-1)
    )
    (fc): Linear(in_features=1536, out_features=2003, bias=True)
  )
)
```

### The LAS performance (CER(%))

`CC` : ClovaCall, `A` : AIhub, `NA` : Noise Augmentation, `SA` : SpecAugment

Dataset         | Pretrain /w    A | ClovaCall only   | ClovaCall /w NA  | ClovaCall /w SA  |
----------------|------------------|------------------|------------------|------------------|
ClovaCall-Base(R)       | 8.0              | 22.1             | -             | -            |
ClovaCall-Full          | 7.0              | 15.1             | 18.9             | 31.1             |


## 4. Dependency

Our code requires the following libraries:
* [PyTorch](https://pytorch.org/)

```
librosa==0.7.0
scipy==1.3.1
numpy==1.17.2
tqdm==4.36.1
torch=1.2.0
python-Levenshtein==0.12.0
```

## 5. Training and Evaluation

Before training or evaluation, we should be follow the data pipeline as the followed.

```
las.pytorch/
run_script/
data/
└──kor_syllable.json
└──ClovaCall/
    └──raw/
        -- 42_0603_748_0_03319_00.wav
            ...
        -- 42_0610_778_0_03607_01.wav
    └──clean/
        -- 42_0603_748_0_03319_00.wav
           ...
        -- 42_0610_778_0_03607_01.wav
    └──train_ClovaCall.json
    └──test_ClovaCall.json
└──Dataset2/
    └──sub1/
        -- audio1.wav
        -- audio2.wav
            ...
    └──sub2/
        -- audio1.wav
        -- audio2.wav
            ...
    └──train_Dataset2.json/
    └──test_Dataset2.json/
```

`--dataset-path` argument can be `data/ClovaCall/raw` or `data/ClovaCall/clean` or `data/Dataset2/sub1`.
`kor_syllable.json` is a json file containing a vocabulary list. Our `kor_syllable.json` is based on character.

Here is a command line example of the code:

```
TRAIN_NAME='train_ClovaCall'
TEST_NAME='test_ClovaCall'
LABEL_FILE=data/kor_syllable.json
DATASET_PATH=data/ClovaCall/clean
CUR_MODEL_PATH=models/ClovaCall/LAS_ClovaCall

python3.6 -u las.pytorch/main.py \
--batch_size 64 \
--num_workers 4 \
--num_gpu 1 \
--rnn-type LSTM \
--lr 3e-4 \
--learning-anneal 1.1 \
--dropout 0.3 \
--teacher_forcing 1.0 \
--encoder_layers 3 --encoder_size 512 \
--decoder_layers 2 --decoder_size 512 \
--train-name $TRAIN_NAME --test-name-list $TEST_NAME \
--labels-path $LABEL_FILE \
--dataset-path $DATASET_PATH \
--cuda --save-folder $CUR_MODEL_PATH --model-path $CUR_MODEL_PATH/final.pth
```

### Run train

```
cd run_script

./run_las_asr_trainer.sh
```

### Run eval
```
cd run_script

./run_las_asr_decode.sh
```

## 6. Code license

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## 7. Reference
* Model/Code
   * IBM pytorch-seq2seq (https://github.com/IBM/pytorch-seq2seq)
   * SeanNaren pytorch-Deepspeech2 (https://github.com/SeanNaren/deepspeech.pytorch)
* Dataset
   * AI Hub open domain dialog speech corpus data: http://www.aihub.or.kr/aidata/105 
   * ClovaAi speech hackathon 2019 data: https://github.com/clovaai/speech_hackathon_2019/tree/master/sample_dataset/train
      * liscense: https://github.com/clovaai/speech_hackathon_2019/blob/master/LICENSE_Data

## 8. How to cite
```
@article{ha2020clovacall,
  title={ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers},
  author={Jung-Woo Ha, Kihyun Nam, Jingu Kang, Sang-Woo Lee, Sohee Yang, Hyunhoon Jung, Eunmi Kim, Hyeji Kim, Soojin Kim, Hyun Ah Kim, Kyoungtae Doh, Chan Kyu Lee, Nako Sung, Sunghun Kim},
  journal={arXiv preprint arXiv:2004.09367},
  year = {2020}
}
```
