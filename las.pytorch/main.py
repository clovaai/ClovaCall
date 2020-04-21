"""
Copyright 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import json
import math
import random
import argparse
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

import Levenshtein as Lev 

import label_loader
from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler

from models import EncoderRNN, DecoderRNN, Seq2Seq


char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 


def get_distance(ref_labels, hyp_labels):
    total_dist = 0
    total_length = 0
    transcripts = []
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])

        transcripts.append('{hyp}\t{ref}'.format(hyp=hyp, ref=ref))

        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 

    return total_dist, total_length, transcripts


def train(model, data_loader, criterion, optimizer, device, epoch, train_sampler, max_norm=400, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.train()
    for i, (data) in enumerate(data_loader):
        feats, scripts, feat_lengths, script_lengths = data

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)
        feat_lengths = feat_lengths.to(device)

        src_len = scripts.size(1)
        target = scripts[:, 1:]

        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)
        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths).item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        dist, length, _ = get_distance(target, y_hat)
        total_dist += dist
        total_length += length
        cer = float(dist / length) * 100

        total_sent_num += target.size(0)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss:.4f}\t'
              'Cer {cer:.4f}'.format(
              (epoch + 1), (i + 1), len(train_sampler), loss=loss, cer=cer))

    return total_loss / total_num, (total_dist / total_length) * 100


def evaluate(model, data_loader, criterion, device, save_output=False):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    transcripts_list = []

    model.eval()
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader)):
            feats, scripts, feat_lengths, script_lengths = data

            feats = feats.to(device)
            scripts = scripts.to(device)
            feat_lengths = feat_lengths.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            logit = model(feats, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            logit = logit[:,:target.size(1),:] # cut over length to calculate loss
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths).item()

            dist, length, transcripts = get_distance(target, y_hat)
            cer = float(dist / length) * 100

            total_dist += dist
            total_length += length
            if save_output == True:
                transcripts_list += transcripts
            total_sent_num += target.size(0)


    aver_loss = total_loss / total_num
    aver_cer = float(total_dist / total_length) * 100
    return aver_loss, aver_cer, transcripts_list


def main():
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='LAS')
    parser.add_argument('--model-name', type=str, default='LAS')
    # Dataset
    parser.add_argument('--train-file', type=str,
                        help='data list about train dataset', default='data/ClovaCall/train_ClovaCall.json')
    parser.add_argument('--test-file-list', nargs='*',
                        help='data list about test dataset', default=['data/ClovaCall/test_ClovCall.json'])
    parser.add_argument('--labels-path', default='data/kor_syllable.json', help='Contains large characters over korean')
    parser.add_argument('--dataset-path', default='data/ClovaCall/clean', help='Target dataset path')
    # Hyperparameters
    parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--encoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of pyramidal layers (default: 2)')
    parser.add_argument('--decoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in training (default: 0.3)')
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True, help='Turn off bi-directional RNNs, introduces lookahead convolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size in training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of gpus (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing learning rate every epoch')
    parser.add_argument('--teacher_forcing', type=float, default=1.0, help='Teacher forcing ratio in decoder (default: 1.0)')
    parser.add_argument('--max_len', type=int, default=80, help='Maximum characters of sentence (default: 80)')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    # Audio Config
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
    parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
    parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')
    # System
    parser.add_argument('--save-folder', default='models', help='Location to save epoch models')
    parser.add_argument('--model-path', default='models/las_final.pth', help='Location to save best validation model')
    parser.add_argument('--log-path', default='log/', help='path to predict log about valid and test dataset')
    parser.add_argument('--cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--mode', type=str, default='train', help='Train or Test')
    parser.add_argument('--load-model', action='store_true', default=False, help='Load model')
    parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
                        help='Finetune the model after load model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    char2index, index2char = label_loader.load_label_json(args.labels_path)
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    device = torch.device('cuda' if args.cuda else 'cpu')

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride)

    # Batch Size
    batch_size = args.batch_size * args.num_gpu

    print(">> Train dataset : ", args.train_file)
    trainData_list = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        trainData_list = json.load(f)

    if args.num_gpu != 1:
        last_batch = len(trainData_list) % batch_size
        if last_batch != 0 and last_batch < args.num_gpu:
            trainData_list = trainData_list[:-last_batch]

    train_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                       dataset_path=args.dataset_path, 
                                       data_list=trainData_list,
                                       char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                       normalize=True)

    train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)


    print(">> Test dataset : ", args.test_file_list)
    testLoader_dict = {}
    for test_file in args.test_file_list:
        testData_list = []
        with open(test_file, 'r', encoding='utf-8') as f:
            testData_list = json.load(f)
        
        test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                          dataset_path=args.dataset_path, 
                                          data_list=testData_list,
                                          char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                          normalize=True)
        testLoader_dict[test_file] = AudioDataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)


    input_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1)
    enc = EncoderRNN(input_size, args.encoder_size, n_layers=args.encoder_layers,
                     dropout_p=args.dropout, bidirectional=args.bidirectional, 
                     rnn_cell=args.rnn_type, variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.decoder_size, args.encoder_size,
                     SOS_token, EOS_token,
                     n_layers=args.decoder_layers, rnn_cell=args.rnn_type, 
                     dropout_p=args.dropout, bidirectional_encoder=args.bidirectional)


    model = Seq2Seq(enc, dec)

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    optim_state = None
    if args.load_model:  # Starting from previous model
        print("Loading checkpoint model %s" % args.model_path)
        state = torch.load(args.model_path)
        model.load_state_dict(state['model'])
        print('Model loaded')

        if not args.finetune:  # Just load model
            optim_state = state['optimizer']
 
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if optim_state is not None:
            optimizer.load_state_dict(optim_state)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    print(model)
    print("Number of parameters: %d" % Seq2Seq.get_param_size(model))

    train_model = nn.DataParallel(model)

    if args.mode != "train":
        for test_file in args.test_file_list:
            test_loader = testLoader_dict[test_file]
            test_loss, test_cer, transcripts_list = evaluate(model, test_loader, criterion, device, save_output=True)

            for line in transcripts_list:
                print(line)

            print("Test {} CER : {}".format(test_file, test_cer))
    else:
        best_cer = 1e10
        begin_epoch = 0
        
        for epoch in range(begin_epoch, args.epochs):
            train_loss, train_cer = train(train_model, train_loader, criterion, optimizer, device, epoch, train_sampler, args.max_norm, args.teacher_forcing)

            cer_list = []
            for test_file in args.test_file_list:
                test_loader = testLoader_dict[test_file]
                test_loss, test_cer, _ = evaluate(model, test_loader, criterion, device, save_output=False)
                test_log = 'Test({name}) Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}\t'.format(
                            epoch + 1, name=test_file, loss=test_loss, cer=test_cer)
                print(test_log)

                cer_list.append(test_cer)

            if best_cer > cer_list[0]:
                print("Found better validated model, saving to %s" % args.model_path)
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, args.model_path)
                best_cer = cer_list[0]

            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

if __name__ == "__main__":
    main()
