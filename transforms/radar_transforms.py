import numpy as np
import torch as th
import cv2

def microdoppler_transform(sample, sample_length=20, features='range_doppler_log', values=None, scaling='standard', preprocessing=False, resized=False):


    # Repeat last frame if less than sample length.
    if len(sample) < sample_length:
        sample = np.concatenate((sample, np.repeat(sample[-1][np.newaxis], sample_length - len(sample), axis=0)))


    if scaling == 'standard':
        sample = (sample - values["mean"]) / (values["std"])
        sample = np.nan_to_num(sample)
    elif scaling == 'minmax':
        sample = (sample - values["min"]) / (values["max"] - values["min"])
    elif scaling == 'local':
        if sample.min() != sample.max():
            sample = (sample - sample.min()) / (sample.max() - sample.min())

    if preprocessing:

        if features[:len('range_doppler')] == 'range_doppler':

            if resized:
                sample = np.asarray([cv2.resize(sample[i], (64, 40)) for i in range(len(sample))])
                sample = np.concatenate((sample[:, :, :32], sample[:, :, 33:]), axis=2)
            else:
                sample = np.asarray([cv2.resize(sample[i], (128, 80)) for i in range(len(sample))])
                sample = np.concatenate((sample[:, :, :63], sample[:, :, 65:]), axis=2)

        else:
            sample = np.concatenate((sample[:, :127], sample[:, 130:]), axis=1)


    return th.from_numpy(sample.astype(np.float32))