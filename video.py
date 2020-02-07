from __future__ import print_function

from models.resnet3d import resnet34
from datasets import VideoDataset, EqualDistSampler
from deploy import train
from utils import load_checkpoint
from torch.utils.data import DataLoader

import os
import torch
import csv
import argparse
import socket
import torch.optim as optim
import torch.nn as nn
import functools

from transforms.image_transforms import video_transform as transform

parser = argparse.ArgumentParser(description='Action Recognition')
parser.add_argument('--params', default='', type=str)
parser.add_argument('--name', default='default', type=str)
parser.add_argument('--dataset', default=['gestures'], nargs='+')
parser.add_argument('--version', default='random', type=str)
parser.add_argument('--version_train', nargs='+')
parser.add_argument('--version_valid', nargs='+')
parser.add_argument('--learning_rate', default=10 ** -4, type=float)
parser.add_argument('--network', default='resnet34', type=str)
parser.add_argument('--sample_size', default=112, type=int)
parser.add_argument('--resize_to', default=128, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--sample_length', default=24, type=int)
parser.add_argument('--max_epochs', default=50, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--location', default='<default>', type=str)
parser.add_argument('--test_location', default='<default>', type=str)
parser.add_argument('--csv_location', default='<default>', type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)

print(args)

# Get accuracte location based on hostname
if args.location == "<default>":
    args.location = '/media/datadrive/baptist/data/autolabel/'
if args.test_location == "<default>":
    args.test_location = '/media/datadrive/baptist/data/autolabel/'
if args.csv_location == "<default>":
    args.csv_location = '/home/baptist/projects/actions/data/csv/'

labels = []
for d in args.dataset:
    with open(os.path.join(args.csv_location, '%s-labels.csv'%d), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        labels += [v for v, in reader]

transform_train = functools.partial(transform,  apply_resize=True,
                                                size=args.sample_size,
                                                resize_to=args.resize_to,
                                                apply_hflip=True,
                                                apply_randomcrop=True,
                                                apply_brightness=True,
                                                apply_saturation=True,
                                                apply_zoom=False,
                                                apply_wb=False)

transform_valid = functools.partial(transform, apply_resize=True,
                                               apply_centercrop=True,
                                               size=args.sample_size,
                                               resize_to=args.sample_size,
                                               apply_wb=False)

transform_test = functools.partial(transform, apply_resize=True,
                                              apply_centercrop=True,
                                              apply_focus=False,
                                              size=args.sample_size,
                                              resize_to=args.sample_size,
                                              apply_wb=False)

# List all files for train, valid, and test set.
train_files = []
valid_files = []
for d in args.dataset:
    if args.version == '':
        train_files += [os.path.join(args.csv_location, '%s-%s.csv' % (d, version)) for version in args.version_train]
        valid_files += [os.path.join(args.csv_location, '%s-%s.csv' % (d, version)) for version in args.version_valid]
    else:
        train_files += [os.path.join(args.csv_location, '%s-%s-train.csv'% (d, args.version))]
        valid_files += [os.path.join(args.csv_location, '%s-%s-validation.csv'%(d, args.version))]

dataset = dict(train=VideoDataset(train_files, labels, select='random', transform=transform_train, sample_length=args.sample_length, sample_size=args.sample_size, location=args.location),
               valid=VideoDataset(valid_files, labels, select='center', transform=transform_valid, sample_length=args.sample_length, sample_size=args.sample_size, location=args.test_location))



sampler = EqualDistSampler(dataset['train']._targets)
train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler) #shuffle=True) #

valid_loader = DataLoader(dataset["valid"], batch_size=args.batch_size, num_workers=args.num_workers)

if args.version == 'random':
    test_files = []
    for d in args.dataset:
        test_files += [
            os.path.join(args.csv_location, '%s-%s-test.csv' % (d, args.version))] if args.version != '' else \
            [os.path.join(args.csv_location, '%s-%s.csv' % (d, version)) for version in args.version_test]
    dataset['test'] = VideoDataset(test_files, labels, select='center', transform=transform_valid, sample_length=args.sample_length, sample_size=args.sample_size, location=args.location)
    test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)


net = eval(args.network)( sample_size=(3, args.sample_length, args.sample_size, args.sample_size), num_classes=len(labels))
net.cuda()

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

if args.params != "":
    load_checkpoint(net, None, args.params, remove_module=True)

criterion = nn.CrossEntropyLoss().cuda()

if not args.test:
    print("---------")
    print("%d samples and %d batches in train set." % (len(dataset['train']), len(train_loader)))
    print("%d samples and %d batches in validation set." % (len(dataset['valid']), len(valid_loader)))
    if args.version == 'random':
        print("%d samples and %d batches in test set." % (len(dataset['test']), len(test_loader)))
    print("---------")

    train(net,
          dict(train=train_loader, valid=valid_loader, test=None if args.version != 'random' else test_loader),
          args.name,
          optimizer=optimizer,
          criterion=criterion,
          max_epochs=args.max_epochs,
          phases=['train', 'valid'] if args.version != 'random' else ['train', 'valid', 'test'],
          classlabels=labels,
    )