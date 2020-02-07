# -*- coding: utf-8 -*-

"""
This code was originally developed for research purposes by IDLab at Ghent University - imec, Belgium.
Its performance may not be optimized for specific applications.

For information on its use, applications and associated permission for use, please contact Baptist Vandersmissen (baptist.vandersmissen@ugent.be).

Detailed information on the activities of Ghent University IDLab can be found at http://idlab.technology/.

Copyright (c) Ghent University - imec 2018-2023.

This code can be used to read and process the all hdf5 files in a certain folder using the multiprocessing library.
It will store the range-doppler maps together with the raw and thresholded microdoppler signature in the same hdf5 file.

Example:    The following line will compute and save the according 'range_doppler' and 'microdoppler' datasets in hdf5 file 'subject1_01.hdf5'.

            python3 process.py --input <path_to_dataset>/subject1_01.hdf5

"""

import numpy as np
import argparse
import glob
import h5py
import multiprocessing as mp
import os


def range_doppler(data, chirps=256,
                  samples=256,
                  fft_rangesamples=2 ** 10,
                  fft_dopplersamples=2 ** 8,
                  fs=2.0e6,
                  kf=1171875.0e7,
                  min_range=0.5,
                  max_range=4.5):
    """
    Computes a range-doppler map for a given number of chirps and samples per chirp.
    :param data: FMCW radar data frame consisting of <chirps>x<samples>
    :param chirps: Number of chirps (Np)
    :param samples: Number of samples (N)
    :param fft_rangesamples: Number of samples for the range fft.
    :param fft_dopplersamples: Number of samples for the doppler fft.
    :param fs: Constant depending on the radar recording parameters.
    :param kf: Constant depending on the radar recording parameters.
    :param min_range: Minimum value to take into account for the range axis in the range-doppler map.
    :param max_range: Maximum value to take into account for the range axis in the range-doppler map.
    :return: Returns a 2D dimensional range-doppler map representing the reflected power over all range-doppler bins.
    """

    data = data.reshape(chirps, samples).T
    # Ignore chirp sequence number
    data = data[1:]
    Ny, Nx = data.shape  # rows (N), columns (Np)

    window = np.hanning(Ny)
    scaled = np.sum(window)
    window2d = np.tile(window, (Nx, 1)).T
    data = data * window2d

    # Calculate Range FFT
    x = np.zeros((fft_rangesamples, Nx))
    start_index = int((fft_rangesamples - Ny) / 2)
    x[start_index:start_index + Ny, :] = data
    X = np.fft.fft(x, fft_rangesamples, 0) / scaled * (2.0 / 2048)
    # Extract positive range bins
    X = X[0:fft_rangesamples // 2, :]
    # Extract range
    _freq = np.arange(fft_rangesamples // 2) / float(fft_rangesamples) * fs
    _range = _freq * 3e8 / (2 * kf)
    min_index = np.argmin(np.abs(_range - min_range))
    max_index = np.argmin(np.abs(_range - max_range))

    X = X[min_index: max_index, :]

    # Calculate Doppler FFT
    Ny, Nx = X.shape
    window = np.hanning(Nx)
    scaled = np.sum(window)
    window2d = np.tile(window, (Ny, 1))
    X = X * window2d

    rd = np.zeros((Ny, fft_dopplersamples), dtype='complex_')
    start_index = int((fft_dopplersamples - Nx) / 2)
    rd[:, start_index:start_index + Nx] = X

    range_doppler = np.fft.fft(rd, fft_dopplersamples, 1) / scaled
    range_doppler = np.fft.fftshift(range_doppler, axes=1)

    return np.abs(range_doppler)


def save_to_file(filename):
    # Read hdf5 file
    with h5py.File(filename, 'r+') as file:
        nframes = file['radar'].shape[0]

        # Create datasets
        if not 'microdoppler' in file:
            file.create_dataset("microdoppler", (nframes, 256), dtype='float32', chunks=(1, 256))
        if not 'range_doppler' in file:
            file.create_dataset("range_doppler", (nframes, 160, 256), dtype='float32', chunks=True)

        # Run over each radar frame
        rds = []
        for i in range(nframes):
            rds.append(range_doppler(file['radar'][i]))

        rds = np.array(rds, copy=False)
        rds = (20 * np.log10(rds))
        file['range_doppler'][:] = rds
        file['microdoppler'][:] = rds.sum(axis=1)
        file.close()

        print("Finished %s." % filename)


if __name__ == '__main__':

    """
    Example reading and processing of all hdf5 files using multiprocessing.
    """

    parser = argparse.ArgumentParser(description='Human Activity Recognition Data Set Script')
    parser.add_argument('--folder', default='./harrad', type=str)
    args = parser.parse_args()

    pool = mp.Pool(processes=12)
    pool.map(save_to_file, glob.glob(os.path.join(args.folder, '*.hdf5')))
    pool.close()
    pool.join()


