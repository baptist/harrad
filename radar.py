from models.models import *
from datasets import RadarDataset, EqualDistSampler
from deploy import train
from utils import load_checkpoint
from torch.utils.data import DataLoader

import os
import socket
import torch
import csv
import sys
import argparse
import torch.optim as optim
import torch.nn as nn
import functools
import numpy as np

from transforms.radar_transforms import microdoppler_transform as transform

parser = argparse.ArgumentParser(description='Action Recognition')
parser.add_argument('--params', default='', type=str)
parser.add_argument('--name', default='default', type=str)
parser.add_argument('--features', default='range_doppler', type=str)
parser.add_argument('--dataset', default=['gestures'], nargs='+')

parser.add_argument('--network', default='CNN3DNet', type=str)
parser.add_argument('--version', default='random', type=str)

parser.add_argument('--version_train', nargs='+')
parser.add_argument('--version_valid', nargs='+')
parser.add_argument('--print', default='', type=str)
parser.add_argument('--scaling', default='minmax', type=str)
parser.add_argument('--learning_rate', default=10 ** -3, type=float)
parser.add_argument('--momentum', default=0., type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--sample_length', default=20, type=int)
parser.add_argument('--max_epochs', default=500, type=int)
parser.add_argument('--resized', action='store_true')
parser.set_defaults(resized=False)
parser.add_argument('--test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--unprocessed', action='store_false', dest='preprocessed')
parser.set_defaults(preprocessed=True)



args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.print != '':
    sys.stdout = open(args.print,'wt')

base_location = '/media/datadrive/baptist/data/harrad/'
csv_location = '/media/datadrive/baptist/data/harrad/labels/'


values = dict()
values['microdoppler'] = {'mean':-18548.79815690202, 'std':1202.8846406929292, 'min':-20102.396484375, 'max':0.0}
values['microdoppler_thresholded'] = {'mean':-10155.406075214181, 'std':751.592937518451, 'min':-10400.0, 'max':0.0}
values['range_doppler'] = {'mean':-115.96167423431216, 'std':47.856299075087996, 'min':-200.0, 'max':-24.600444793701172}
values['range_doppler_thresholded'] = {'mean':-63.88854599273084, 'std':12.314452167197178, 'min':-65.0, 'max':0}

if not args.resized:
    sample_sizes = {
        'microdoppler': {0: (1, args.sample_length, 256), 1: (1, args.sample_length, 253)},
        'range_doppler': {0: (1, args.sample_length, 80, 128), 1: (1, args.sample_length, 80, 126)},
    }
else:
    sample_sizes = {
        'microdoppler': {0: (1, args.sample_length, 256), 1: (1, args.sample_length, 253)},
        'range_doppler': {0: (1, args.sample_length, 40, 64), 1: (1, args.sample_length, 40, 63)},
    }

transform= functools.partial(transform,
                             sample_length=args.sample_length,
                             features=args.features,
                             scaling=args.scaling,
                             resized=args.resized,
                             preprocessing=args.preprocessed,
                             values=values[args.features])#None)#

labels = []
for d in args.dataset:
    with open(os.path.join(csv_location, '%s-labels.csv'%d), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        labels += [v for v, in reader]

# List all files for train, valid, and test set.
train_files = []
valid_files = []
for d in args.dataset:
    if args.version == '':
        train_files += [os.path.join(csv_location, '%s-%s.csv' % (d, version)) for version in args.version_train]
        valid_files += [os.path.join(csv_location, '%s-%s.csv' % (d, version)) for version in args.version_valid]
    else:
        train_files += [os.path.join(csv_location, '%s-%s-train.csv'% (d, args.version))]
        valid_files += [os.path.join(csv_location, '%s-%s-validation.csv'%(d, args.version))]

has_test = (args.version == 'random')

print(labels)

dataset = dict(train=RadarDataset(train_files, labels,
                                  transform=transform,
                                  feature=args.features,
                                  in_memory=(args.features!='range_doppler'),
                                  sample_length=args.sample_length,
                                  base_location=base_location),

                valid=RadarDataset(valid_files, labels,
                                   transform=transform,
                                   feature=args.features,
                                   select='center',
                                   in_memory=True,
                                   sample_length=args.sample_length,
                                   base_location=base_location))

sampler = EqualDistSampler(dataset['train']._targets)
train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, num_workers=12, sampler=sampler)

valid_loader = DataLoader(dataset["valid"], batch_size=args.batch_size, num_workers=12)


if has_test:

    test_files = []
    for d in args.dataset:
        test_files += [
            os.path.join(csv_location, '%s-%s-test.csv' % (d, args.version))] if args.version != '' else \
            [os.path.join(csv_location, '%s-%s.csv' % (d, version)) for version in args.version_test]

    dataset['test'] = RadarDataset(test_files, labels,
                                   transform=transform,
                                   feature=args.features,
                                   select='center',
                                   in_memory=True,
                                   sample_length=args.sample_length,
                                   base_location=base_location)
    test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, num_workers=12)

net = eval(args.network)(num_classes=len(labels), sample_size=sample_sizes[args.features][int(args.preprocessed)])
net.cuda()

if args.params != "":
    load_checkpoint(net, None, args.params)

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss().cuda()

if not args.test:

    print(args)
    print("---------")
    print("%d samples and %d batches in train set." % (len(dataset['train']), len(train_loader)))
    print("%d samples and %d batches in validation set." % (len(dataset['valid']), len(valid_loader)))

    if has_test:
        print("%d samples and %d batches in test set." % (len(dataset['test']), len(test_loader)))
    print("---------")

    train(net,
          dict(train=train_loader, valid=valid_loader, test=None if not has_test else test_loader),
          args.name,
          optimizer=optimizer,
          criterion=criterion,
          max_epochs=args.max_epochs,
          phases=['train', 'valid'] if not has_test else ['train', 'valid', 'test'],
          classlabels=labels)