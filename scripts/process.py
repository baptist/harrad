# -*- coding: utf-8 -*-

"""
This code was originally developed for research purposes by IDLab at Ghent University - imec, Belgium.
Its performance may not be optimized for specific applications.

For information on its use, applications and associated permission for use, please contact Baptist Vandersmissen (baptist.vandersmissen@ugent.be).

Detailed information on the activities of Ghent University IDLab can be found at http://idlab.technology/.

Copyright (c) Ghent University - imec 2018-2023.

This code can be used to read and process the raw radar data saved in an hdf5 file.
It will store the range-doppler maps together with the raw and thresholded microdoppler signature in the same hdf5 file.

Example:    The following line will compute and save the according 'range_doppler' and 'microdoppler' datasets in hdf5 file 'subject1_01.hdf5'.

            python3 process.py --input <path_to_dataset>/subject1_01.hdf5

"""

import numpy as np
import argparse
import h5py


def range_doppler(data, chirps=256,
                  samples=256,
                  fft_rangesamples=2 ** 9,
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



def doppler_time(data,
                 Np=128,
                 N=256,
                 fft_rangesamples=2 ** 9,
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

    # Speed of light
    c = 3e8
    data_ = np.zeros((len(data), 256, 512), dtype='complex_')
    Freq = np.arange(int(fft_rangesamples)) / fft_rangesamples * fs
    __rangeBins = Freq * c / (2 * kf)
    __numFFTs = 4
    __frameHistoryLength = len(data)

    numRangeBins = len(__rangeBins)
    print(numRangeBins)

    # Compute range
    for i in range(len(data)):
        d = data[i].reshape(256, 256).T
        # Ignore chirp sequence number
        d = d[1:]
        Ny, Nx = d.shape  # rows (N), columns (Np)

        window = np.hanning(Ny)
        scaled = np.sum(window)
        window2d = np.tile(window, (Nx, 1)).T
        d = d * window2d

        # Calculate Range FFT
        x = np.zeros((fft_rangesamples, Nx))
        start_index = int((fft_rangesamples - Ny) / 2)
        x[start_index:start_index + Ny, :] = d
        o = np.fft.fft(x, fft_rangesamples, 0) / scaled * (2.0 / 2048)
        data_[i] = o.T

    # bins = np.arange(-Np // 2, Np // 2)
    # fc = 77e9
    # __dopplerBins = bins * c / (Np * 0.000256 * 2 * fc)
    # __inputFrameSlices = (slice(Np), slice(numRangeBins))

    __numRangeGroups = numRangeBins


    __spectrogramLength = __frameHistoryLength * __numFFTs - 1
    __spectrogramBuffer = np.empty((__spectrogramLength, Np, __numRangeGroups), dtype=float)
    __spectrogramBufferPos = 0
    # The window has to be applied to the first dimension only, so we
    # reshape it with unit sizes for the other 3 dimensions, in order to
    # make use of broadcasting during multiplication.
    __spectrogramWindow = np.hanning(Np).reshape((-1, 1))

    for i in range(len(data)):

        rangeFFT = data_[i]

        # Normalize the FFT
        meanPerRange = np.mean(rangeFFT, -1, keepdims=True)
        variancePerRange = np.var(rangeFFT, -1, keepdims=True)
        if (variancePerRange > 0).all():
            rangeFFT = (rangeFFT - meanPerRange) / np.sqrt(variancePerRange)

        # Perform numFFTs FFTs on the last sample, to update the spectrogram
        # (shifting window over both samples, ending with an FFT over only the
        # second sample). The very first FFT will be bogus (because there is
        # no previous sample), but the results are shifted out of the buffer
        # before we use them, so there is no need to treat this case
        # specially.

        for pos in range(__numFFTs):
            sliceStart = (pos + 1) * Np // __numFFTs
            sliceEnd = sliceStart + Np
            fftSlice = rangeFFT[sliceStart:sliceEnd]
            microDopplerFFT = np.fft.fft(fftSlice * __spectrogramWindow, Np, axis=0)

            # Set the DC component to zero.
            # microDopplerFFT[0] = 0
            # microDopplerFFT[1] = 0
            # microDopplerFFT[-1] = 0
            __spectrogramBuffer[__spectrogramBufferPos] = np.abs(microDopplerFFT)
            __spectrogramBufferPos = (__spectrogramBufferPos + 1) % __spectrogramLength

    rgStart = np.argmin(np.abs(__rangeBins - min_range))
    rgEnd = np.argmin(np.abs(__rangeBins - max_range))


    # Sum the spectrograms over the range window
    microDoppler = np.sum(__spectrogramBuffer[:, :, rgStart:rgEnd], axis=-1)

    # FFTs are not centered -> rotate the values.
    microDoppler = np.fft.fftshift(microDoppler, axes=(1,))

    # Finally, rotate the microDoppler columns, to take into account
    # the sliding effect of the spectrogram buffer.
    # microDoppler = np.roll(microDoppler, -__spectrogramBufferPos, axis=0)


    return microDoppler  # Traditional order


if __name__ == '__main__':

    """
    Example reading and processing of hdf5 file.
    This script will add two datasets to the hdf5 file, namely 'range_doppler' and 'microdoppler'.
    """

    parser = argparse.ArgumentParser(description='Human Activity Recognition Data Set Script')
    parser.add_argument('--input', default='subject1_01.hdf5', type=str)
    args = parser.parse_args()

    # Read hdf5 file
    with h5py.File(args.input, 'r+') as file:
        nframes = file['radar'].shape[0]

        # Create datasets
        if not 'microdoppler' in file:
            file.create_dataset("microdoppler", (nframes, 256), dtype='float32', chunks=(1, 256))
        if not 'range_doppler' in file:
            file.create_dataset("range_doppler", (nframes, 160, 256), dtype='float32', chunks=True)

        # Run over each radar frame
        for i in range(nframes):
            rd = range_doppler(file['radar'][i])
            # md = doppler_time(file['radar'][i])
            file['microdoppler'][i] = rd.sum(axis=0)


            print("Finished frame %d of %d." % (i + 1, nframes))
