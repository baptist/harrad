import os
import numpy as np
import cv2
import torch
import random
import csv
import h5py

from os.path import join
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class VideoDataset(Dataset):

    def __init__(self, samples_or_files, labels, transform, sample_length=12, num_fragments=1, sample_size=100, select='random', location=''):

        self._data = []
        self._targets = []
        self._num_frms = []
        self._exts = []
        self._sample_length = sample_length
        self._location = location
        self._select = select
        self._transform = transform
        self._num_fragments = num_fragments
        self._labels = labels
        self._sample_size = sample_size

        samples = []
        for sample_file in samples_or_files:
            with open(sample_file, 'r') as f:
                samples += [data for data in csv.reader(f, delimiter=',')]

        for data in samples:
            filename, start_index, end_index, label, person = data
            start_index = int(start_index)
            end_index = int(end_index)

            self._data.append((filename, start_index, end_index))
            self._num_frms.append(max(1, end_index - start_index))
            self._targets.append(labels.index(label))

            self._exts.append([file for file in os.listdir(join('' if filename[0] == '/' else self._location, filename))])

        self._max_length = np.max(self._num_frms)

    def __getitem__(self, index):

        if self._select == 'center':
            begin_index = max(0, (self._num_frms[index] - self._sample_length) // 2)
            shift = 0
        elif self._select == 'random':
            begin_index = np.random.randint(0, max(1, self._num_frms[index] - self._sample_length))
            shift = 0
        elif self._select == 'uniform':
            begin_index = 0
            shift = max(1, (self._num_frms[index] - self._sample_length) // self._num_fragments)
        elif self._select == 'action':
            if len(self._poi[index]) > 0:
                margin = min(1, int(self._sample_length * .1))
                poi = self._poi[index][np.random.randint(0, len(self._poi[index]))][0]

                if poi - margin > max(0, poi + margin - self._sample_length):
                    begin_index = np.random.randint(max(0, poi + margin - self._sample_length), poi - margin)
                else:
                    begin_index = max(0, poi + margin - self._sample_length)

                if begin_index + self._sample_length > self._num_frms[index]:
                    begin_index = max(0, self._num_frms[index] - self._sample_length)

                shift = 0

            else:
                begin_index = np.random.randint(0, max(1, self._num_frms[index] - self._sample_length))
                shift = 0

        if self._sample_length == -1:
            begin_index = 0
            shift = 0
            length = self._num_frms[index]
            sample = torch.zeros(self._num_fragments, self._max_length, 3, self._sample_size, self._sample_size)
        else:
            length = self._sample_length
            sample = torch.zeros(self._num_fragments, self._sample_length, 3, self._sample_size, self._sample_size)

        for h in range(self._num_fragments):
            imgs = []

            for i in range(min(length, self._num_frms[index] - h * shift - begin_index)):
                imgs.append(cv2.imread(join(self._location if not self._data[index][0][0] == '/' else '', self._data[index][0], str(self._data[index][1] + begin_index + (h * shift) + i + 1).zfill(5) + ('.' if self._select != 'random' else '.') + self._exts[index]))[:, :, ::-1])

            if self._transform is not None:
                sample[h, :len(imgs), :3] = self._transform(np.asarray(imgs), self._labels[self._targets[index]])
            else:
                sample = np.asarray(imgs)

            # Repeat last frame
            if len(imgs) < self._sample_length:
                sample[h, len(imgs):] = sample[h, len(imgs) - 1].unsqueeze(0).repeat(self._sample_length - len(imgs), 1, 1, 1)

        return sample.squeeze(), self._targets[index] if self._targets[index] >= 0 else self._data[index][0]


    def __len__(self):
        return len(self._data)




class RadarDataset(Dataset):

    def __init__(self, samples_or_files, labels, transform, feature='microdoppler', sample_length=30, select='random', margin=0, base_location='./harrad', in_memory=True):

        self._data = []
        self._targets = []
        self._num_frms = []
        self._info = []
        self._sample_length = sample_length
        self._transform = transform
        self._labels = labels
        self._feature = feature
        self._select = select
        self._margin = margin
        self._in_memory = in_memory

        samples = []
        for sample_file in samples_or_files:
            with open(sample_file, 'r') as f:
                samples += [data for data in csv.reader(f, delimiter=',')]

        for data in samples:

            filename, start_index, end_index, label, person = data
            start_index = int(start_index)
            end_index = int(end_index)

            # Ignore sample if label is not in set of labels
            if label not in labels:
                continue

            with h5py.File(os.path.join(base_location, filename), 'r') as file:
                # self._data.append((file[self._feature][start_index*256: end_index*256, 1:].reshape((-1, 256*255)), labels.index(label)))

                try:
                    if self._in_memory:
                        self._data.append((file[self._feature][start_index: end_index], labels.index(label)))

                    self._num_frms.append(end_index - start_index)
                    self._targets.append(labels.index(label))
                    self._info.append((os.path.join(base_location, filename), start_index, end_index, label))
                except ValueError as e:
                    print(e)
                    print(file[self._feature].shape, file['radar'].shape,
                          file[self._feature][start_index: end_index].shape, start_index, end_index)

        self._max_length = np.max(self._num_frms)

    def __getitem__(self, index):

        if self._select == 'center':
            begin_index = max(0, (self._num_frms[index] - self._sample_length) // 2)
        else:
            begin_index = np.random.randint(0, max(1, self._num_frms[index] - self._sample_length))

        if self._in_memory:
            sample, label = self._data[index]
        else:
            filename, start_index, end_index, _ = self._info[index]
            with h5py.File(filename, 'r') as file:
                sample = file[self._feature][start_index:end_index]
            label = self._targets[index]

        sample = sample[begin_index:begin_index+self._sample_length]

        if self._transform is not None:
            sample = self._transform(sample)

        return sample.unsqueeze(0), label


    def __len__(self):
        return len(self._info)



class EqualDistSampler(Sampler):
    """
        label shuffling technique aimed to deal with imbalanced class problem
        without replacement, manipulated by indices.
        All classes are enlarged to the same amount, so classes can be trained equally.
        argument:
        indices: indices of labels of the whole dataset
        """

    def __init__(self, indices):
        # mapping between label index and sorted label index
        sorted_labels = sorted(enumerate(indices), key=lambda x: x[1])
        count = 1
        count_of_each_label = []
        tmp = -1
        # get count of each label
        for (x, y) in sorted_labels:
            if y == tmp:
                count += 1
            else:
                if tmp != -1:
                    count_of_each_label.append(count)
                    count = 1
            tmp = y
        count_of_each_label.append(count)
        # get the largest count among all classes. used to enlarge every class to the same amount
        largest = max(count_of_each_label)
        self.count_of_each_label = count_of_each_label
        self.enlarged_index = []

        # preidx used for find the mapping beginning of arg "sorted_labels"
        preidx = 0
        for x in range(len(self.count_of_each_label)):
            idxes = torch.remainder(torch.randperm(largest), self.count_of_each_label[x]) + preidx
            for y in idxes:
                self.enlarged_index.append(sorted_labels[y][0])
            preidx += int(self.count_of_each_label[x])

    def __iter__(self):
        random.shuffle(self.enlarged_index)
        return iter(self.enlarged_index)

    def __len__(self):
        return max(self.count_of_each_label) * len(self.count_of_each_label)