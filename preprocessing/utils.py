import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from tqdm import tqdm


class Standardizer:
    """"""
    def __init__(self, model_fn=None):
        """"""
        if model_fn is None:
            # init new standardizer for training
            self._sclr = StandardScaler()
        else:
            self._sclr = joblib.load(model_fn)

    def fit(self, mel_fns):
        """"""
        for fn in tqdm(mel_fns, ncols=80):
            X = np.load(fn) # (n_steps, n_freq)
            self._sclr.partial_fit(X)

    def transform(self, X):
        """"""
        return self._sclr.transform(X)

    def transform_batch(self, X):
        """
        X (ndarray): batch of mel spectrogram (n_batch, n_ch, n_steps, n_freq)
        """
        X_ = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[3]))
        Y_ = self._sclr.transform(X_)
        Y = np.reshape(Y_, X.shape)
        return Y

    def save(self, fn):
        """"""
        joblib.dump(self._sclr, fn)


def train_sclr(mel_dir, sclr_out_fn):
    """"""
    fns = glob.glob(os.path.join(mel_dir, '*.npy'))

    sclr = Standardizer()
    sclr.fit(fns)
    sclr.save(sclr_out_fn)

if __name__ == "__main__":
    fire.Fire(train_sclr)
