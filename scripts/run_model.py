# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import random
import shutil
import sys
import os

#import plotly.offline as py
#import plotly.graph_objs as go

import seaborn
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize, imsave

import scipy.interpolate as ip

# import cv2

import iep.utils as utils
import iep.programs
from iep.data import ClevrDataset, ClevrDataLoader
from iep.preprocess import tokenize, encode


parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_question_h5', default='data/val_questions.h5')
parser.add_argument('--input_features_h5', default='data-ssd/val_features.h5')
parser.add_argument('--use_gt_programs', default=0, type=int)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--image', default=None)
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)

parser.add_argument('--focus_data', default=None)
parser.add_argument('--focus_img', default=None)

def init_model(args):
  model = None
  if args.baseline_model is not None:
    print('Loading baseline model from ', args.baseline_model)
    model, _ = utils.load_baseline(args.baseline_model)
    if args.vocab_json is not None:
      new_vocab = utils.load_vocab(args.vocab_json)
      model.rnn.expand_vocab(new_vocab['question_token_to_idx'])
  elif args.program_generator is not None and args.execution_engine is not None:
    print('Loading program generator from ', args.program_generator)
    program_generator, _ = utils.load_program_generator(args.program_generator)
    print('Loading execution engine from ', args.execution_engine)
    execution_engine, _ = utils.load_execution_engine(args.execution_engine, verbose=False)
    if args.vocab_json is not None:
      new_vocab = utils.load_vocab(args.vocab_json)
      program_generator.expand_encoder_vocab(new_vocab['question_token_to_idx'])
    model = (program_generator, execution_engine)
  else:
    print('Must give either --baseline_model or --program_generator and --execution_engine')
    return None
  return model


def main(args):
  print()
  model = init_model(args)
  if model is None:
    return

  if args.question is not None and args.image is not None:
    run_single_example(args, model)
  else:
    vocab = load_vocab(args)
    loader_kwargs = {
      'question_h5': args.input_question_h5,
      'feature_h5': args.input_features_h5,
      'vocab': vocab,
      'batch_size': args.batch_size,
    }
    if args.num_samples is not None and args.num_samples > 0:
      loader_kwargs['max_samples'] = args.num_samples
    if args.family_split_file is not None:
      with open(args.family_split_file, 'r') as f:
        loader_kwargs['question_families'] = json.load(f)
    with ClevrDataLoader(**loader_kwargs) as loader:
      run_batch(args, model, loader)


def load_vocab(args):
  path = None
  if args.baseline_model is not None:
    path = args.baseline_model
  elif args.program_generator is not None:
    path = args.program_generator
  elif args.execution_engine is not None:
    path = args.execution_engine
  return utils.load_cpu(path)['vocab']


def run_single_example(args, model, cnn_in=None):
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor

  # Build the CNN to use for feature extraction
  if cnn_in is None:
    print('Loading CNN for feature extraction')
    cnn = build_cnn(args, dtype)
  else:
    cnn = cnn_in

  # Load and preprocess the image
  img_size = (args.image_height, args.image_width)
  # print(img_size)
  img = imread(args.image, mode='RGB')
  img = imresize(img, img_size, interp='bicubic')
  imsave("resized.png", img)
  img_hm = img
  img = img.transpose(2, 0, 1)[None]
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
  img = (img.astype(np.float32) / 255.0 - mean) / std

  # Use CNN to extract features for the image
  img_var = Variable(torch.FloatTensor(img).type(dtype), volatile=False, requires_grad=True)
  feats_var = cnn(img_var)
  # print(feats_var)

  # Tokenize the question
  vocab = load_vocab(args)
  question_tokens = tokenize(args.question,
                      punct_to_keep=[';', ','],
                      punct_to_remove=['?', '.'])
  question_encoded = encode(question_tokens,
                       vocab['question_token_to_idx'],
                       allow_unk=True)
  question_encoded = torch.LongTensor(question_encoded).view(1, -1)
  question_encoded = question_encoded.type(dtype).long()
  question_var = Variable(question_encoded, volatile=False)

  # Run the model
  print('Running the model\n')
  scores = None
  predicted_program = None

  GMAP_W = 14
  GMAP_H = 14

  IMG_W = 320
  IMG_H = 240

  # gm_ffm = [[0 for j in range(GMAP_W)] for i in range(GMAP_H)]
  # gm_cnn = [[0 for j in range(GMAP_W)] for i in range(GMAP_H)]
  gm_ffm = np.zeros([GMAP_H, GMAP_W])

  def hook(gmap, layers, grad):
    # print(grad)
    data = grad.data.cpu()
    maxvalue = 0
    for i in range(GMAP_H):
      for j in range(GMAP_W):
        for k in range(layers):
          gmap[i][j] = gmap[i][j] + data[0][k][i][j]
        if abs(gmap[i][j]) > maxvalue:
          maxvalue = abs(gmap[i][j])
    # print("maxvalue=", maxvalue)
    for i in range(GMAP_H):
      for j in range(GMAP_W):
        gmap[i][j] = abs(gmap[i][j] / maxvalue)
  
  if type(model) is tuple:
    program_generator, execution_engine = model
    program_generator.type(dtype)
    execution_engine.type(dtype)
    predicted_program = program_generator.reinforce_sample(
                          question_var,
                          temperature=args.temperature,
                          argmax=(args.sample_argmax == 1))
    scores, ffm = execution_engine(feats_var, predicted_program)
    ffm.register_hook(lambda grad: hook(gm_ffm, 128, grad))

  else:
    model.type(dtype)
    scores = model(question_var, feats_var)
    feats_var.register_hook(lambda grad: hook(gm_ffm, 1024, grad))

  print("SCORES=", scores[0][0])

  # fv = feats_var.transpose(1, 3)
  # print(fv)

  # feats_var.register_hook(lambda grad: hook(gm_cnn, 1024, grad))
  # fv[0][0][0].sum().backward()
  
  sum = scores.sum()
  sum.backward()

  x = np.zeros([GMAP_H, GMAP_W])
  y = np.zeros([GMAP_H, GMAP_W])
  # # x = [(i + 0.5) / GMAP_H * IMG_H for i in range(GMAP_H)]
  # # y = [(i + 0.5) / GMAP_H * IMG_W for i in range(GMAP_W)]

  z = np.zeros([GMAP_H, GMAP_W])
 
  for i in range(GMAP_H):
    for j in range(GMAP_W):
      x[i][j] = (i) #/ (GMAP_H-1)
      y[i][j] = (j) #/ (GMAP_W-1)
      z[i][j] = gm_ffm[i][j]
  #print(x.max())
  #print(y.max())

  x.reshape([-1])
  y.reshape([-1])
  z.reshape([-1])
  
  # # x_new = np.zeros([IMG_H * IMG_W])
  # # y_new = np.zeros([IMG_H * IMG_W])

  x_new = [i + 0.5 for i in range(IMG_H)]
  y_new = [i + 0.5 for i in range(IMG_W)]

  # # for i in range(IMG_H):
  # #   for j in range(IMG_W):
  # #     x_new[i * IMG_W + j] = i + 0.5
  # #     y_new[i * IMG_W + j] = j + 0.5 
  # # print(x_new.size)
  #x,y=np.mgrid(0:IMG_W:14j,0:IMG_H:14j)

  f = ip.RectBivariateSpline([(i+0.5) / (GMAP_H) for i in range(GMAP_H)], [(i+0.5) / (GMAP_W) for i in range(GMAP_W)], z)
  # f = ip.interp2d(x, y, z, kind='linear', fill_value=0, bounds_error=True)
  # z_new = f(x_new, y_new)
  z_new = np.zeros([IMG_H, IMG_W])
  for i in range(IMG_H):  
    for j in range(IMG_W):
      z_new[i][j] = f((i + 0.5) / IMG_H, (j+0.5) / IMG_W)[0]
  
  if args.focus_data is not None:
    with open(args.focus_data, 'w') as f:
      for row in z_new:
        for d in row:
          f.write(str(d))
          f.write(' ')
        f.write('\n')

  fimg = np.zeros([IMG_H, IMG_W, 3])
  for i in range(IMG_H):
    for j in range(IMG_W):
      val = z_new[i][j] * 255
      fimg[i][j] = [val, val, val]
  if args.focus_img is not None:
    imsave(args.focus_img, fimg)
  # tck = ip.bisplrep(x, y, z, s=0)
  # z_new = ip.bisplev(x_new, y_new, tck)

  #z_new.reshape([IMG_H, IMG_W])
  #print(z.max())
  #print(z_new.max())
  # print(z_new)

  # plt.pcolor(z)
  # plt.show()
  # plt.pcolor(z_new)
  # plt.show()

  # print(gm_ffm)

  # himg = np.array(gmap, dtype=np.float32)
  # himg = imresize(himg, img_size, interp='bicubic')
  # img_ffm = np.zeros([args.image_height, args.image_width, 3])
  # img_cnn = np.zeros([args.image_height, args.image_width, 3])

  # scale_h = args.image_height / GMAP_H
  # scale_w = args.image_width / GMAP_W
  # for i in range(args.image_height):
  #   for j in range(args.image_width):
  #     for k in range(3):
  #       img_ffm[i][j][k] = img_hm[i][j][k] * 0.5 + gm_ffm[int(i / scale_h)][int(j / scale_w)] * 255 * 0.5
  #       img_cnn[i][j][k] = img_hm[i][j][k] * 0.5 + gm_cnn[int(i / scale_h)][int(j / scale_w)] * 255 * 0.5
  
  # imsave('heatmap-ffm.png', img_ffm)
  # imsave('heatmap-cnn.png', img_cnn)

  # print(himg)

  # seaborn.heatmap(gm_ffm)
  # plt.show()

  # print("GRADIENT=", ffm.grad)

  # print(scores.backward(ffm))

  # Print results
  _, predicted_answer_idx = scores.data.cpu()[0].max(dim=0)
  predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx[0]]

  print('Question: "%s"' % args.question)
  print('Predicted answer: ', predicted_answer)

  if predicted_program is not None:
    print()
    print('Predicted program:')
    program = predicted_program.data.cpu()[0]
    num_inputs = 1
    for fn_idx in program:
      fn_str = vocab['program_idx_to_token'][fn_idx]
      num_inputs += iep.programs.get_num_inputs(fn_str) - 1
      print(fn_str)
      if num_inputs == 0:
        break


def build_cnn(args, dtype):
  if not hasattr(torchvision.models, args.cnn_model):
    raise ValueError('Invalid model "%s"' % args.cnn_model)
  if not 'resnet' in args.cnn_model:
    raise ValueError('Feature extraction only supports ResNets')
  whole_cnn = getattr(torchvision.models, args.cnn_model)(pretrained=True)
  layers = [
    whole_cnn.conv1,
    whole_cnn.bn1,
    whole_cnn.relu,
    whole_cnn.maxpool,
  ]
  for i in range(args.cnn_model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(whole_cnn, name))
  cnn = torch.nn.Sequential(*layers)
  cnn.type(dtype)
  cnn.eval()
  return cnn


def run_batch(args, model, loader):
  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor
  if type(model) is tuple:
    program_generator, execution_engine = model
    run_our_model_batch(args, program_generator, execution_engine, loader, dtype)
  else:
    run_baseline_batch(args, model, loader, dtype)


def run_baseline_batch(args, model, loader, dtype):
  model.type(dtype)
  model.eval()

  all_scores, all_probs = [], []
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, images, feats, answers, programs, program_lists = batch

    questions_var = Variable(questions.type(dtype).long(), volatile=True)
    feats_var = Variable(feats.type(dtype), volatile=True)
    scores = model(questions_var, feats_var)
    probs = F.softmax(scores)

    _, preds = scores.data.cpu().max(1)
    all_scores.append(scores.data.cpu().clone())
    all_probs.append(probs.data.cpu().clone())

    num_correct += (preds == answers).sum()
    num_samples += preds.size(0)
    print('Ran %d samples' % num_samples)

  acc = float(num_correct) / num_samples
  print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

  all_scores = torch.cat(all_scores, 0)
  all_probs = torch.cat(all_probs, 0)
  if args.output_h5 is not None:
    print('Writing output to %s' % args.output_h5)
    with h5py.File(args.output_h5, 'w') as fout:
      fout.create_dataset('scores', data=all_scores.numpy())
      fout.create_dataset('probs', data=all_probs.numpy())


def run_our_model_batch(args, program_generator, execution_engine, loader, dtype):
  program_generator.type(dtype)
  program_generator.eval()
  execution_engine.type(dtype)
  execution_engine.eval()

  all_scores, all_programs = [], []
  all_probs = []
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, images, feats, answers, programs, program_lists = batch

    questions_var = Variable(questions.type(dtype).long(), volatile=True)
    feats_var = Variable(feats.type(dtype), volatile=True)

    programs_pred = program_generator.reinforce_sample(
                        questions_var,
                        temperature=args.temperature,
                        argmax=(args.sample_argmax == 1))
    if args.use_gt_programs == 1:
      scores = execution_engine(feats_var, program_lists)
    else:
      scores = execution_engine(feats_var, programs_pred)
    probs = F.softmax(scores)

    _, preds = scores.data.cpu().max(1)
    all_programs.append(programs_pred.data.cpu().clone())
    all_scores.append(scores.data.cpu().clone())
    all_probs.append(probs.data.cpu().clone())

    num_correct += (preds == answers).sum()
    num_samples += preds.size(0)
    print('Ran %d samples' % num_samples)

  acc = float(num_correct) / num_samples
  print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

  all_scores = torch.cat(all_scores, 0)
  all_probs = torch.cat(all_probs, 0)
  all_programs = torch.cat(all_programs, 0)
  if args.output_h5 is not None:
    print('Writing output to "%s"' % args.output_h5)
    with h5py.File(args.output_h5, 'w') as fout:
      fout.create_dataset('scores', data=all_scores.numpy())
      fout.create_dataset('probs', data=all_probs.numpy())
      fout.create_dataset('predicted_programs', data=all_programs.numpy())


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
