"""
"""

import six
import logging

import scipy as sp
import scipy.special  # noqa

import numpy as np

import zcode.math as zmath

__all__ = ['KDE']


class KDE(object):
    """
    """
    bw_method_default = 'scott'

    def __init__(self, dataset, bw_method=None, weights=None, neff=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.ndims, self.nvals = self.dataset.shape
        if weights is None:
            weights = np.ones(self.nvals)/self.nvals

        if weights is not None:
            if np.count_nonzero(weights) == 0 or np.any(~np.isfinite(weights) | (weights < 0)):
                logging.error("weights = '{}'".format(zmath.stats_str(weights)))
                raise ValueError("Invalid `weights` entries, all must be finite and > 0!")
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= np.sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.nvals:
                raise ValueError("`weights` input should be of length n")

        self._neff = neff
        self._weights = weights
        self._compute_covariance()
        self.set_bandwidth(bw_method=bw_method)
        return

    def pdf(self, points):
        points = np.atleast_2d(points)

        ndim, nv = points.shape
        if ndim != self.ndims:
            if ndim == 1 and nv == self.ndims:
                # points was passed in as a row vector
                points = np.reshape(points, (self.ndims, 1))
                nv = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (ndim, self.ndims)
                raise ValueError(msg)

        result = np.zeros((nv,), dtype=float)

        # Compute Cholesky matrix from the covariance matrix, this matrix will transform the
        # dataset to one with a diagonal covariance matrix (i.e. independent variables)
        # note: `sp.linalg.cholesky` uses an *upper* matrix by default, and numpy uses *lower*
        whitening = sp.linalg.cholesky(self.bw_cov_inv)
        # Construct the 'whitened' (independent) dataset
        scaled_dataset = np.dot(whitening, self.dataset)
        # Construct the whitened sampling points
        scaled_points = np.dot(whitening, points)

        # Evaluate kernel at the target sample points

        # Determine the smaller dataset to loop over
        if nv >= self.nvals:
            # there are more points than data, so loop over data
            for i in range(self.nvals):
                diff = scaled_dataset[:, i, np.newaxis] - scaled_points
                energy = np.sum(diff * diff, axis=0) / 2.0
                result += self.weights[i]*np.exp(-energy)
        else:
            # loop over points
            for i in range(nv):
                diff = scaled_dataset - scaled_points[:, i, np.newaxis]
                energy = np.sum(diff * diff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy)*self.weights, axis=0)

        result = result / self.bw_norm

        return result

    def resample(self, size=None, keep=None):
        if size is None:
            size = int(self.neff)

        bw_cov = np.array(self.bw_cov)
        if keep is not None:
            keep = np.atleast_1d(keep)
            for pp in keep:
                bw_cov[pp, :] = 0.0
                bw_cov[:, pp] = 0.0

        # Draw from the smoothing kernel
        # here the `cov` includes the bandwidth
        try:
            norm = np.random.multivariate_normal(np.zeros(self.ndims), bw_cov, size=size)
        except ValueError as err:
            logging.error("Failed to construct `multivariate_normal`!  " + str(err))
            logging.error("cov = '{}'".format(self.bw_cov))
            logging.error("dataset = '{}'".format(zmath.stats_str(self.dataset)))
            raise

        norm = np.transpose(norm)
        # Draw randomly from the given data points, proportionally to their weights
        indices = np.random.choice(self.nvals, size=size, p=self.weights)
        means = self.dataset[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm

        return samps

    def scott_factor(self, *args, **kwargs):
        return np.power(self.neff, -1./(self.ndims+4))

    def silverman_factor(self, *args, **kwargs):
        return np.power(self.neff*(self.ndims+2.0)/4.0, -1./(self.ndims+4))

    def set_bandwidth(self, bw_method=None):
        ndims = self.ndims
        _bw_method = bw_method
        bw_white = np.zeros((ndims, ndims))

        if len(np.atleast_1d(bw_method)) == 1:
            _bw, bw_method = self._compute_bandwidth(bw_method)
            bw_white[...] = _bw
        else:
            if np.shape(bw_method) == (ndims,):
                # bw_method = 'diagonal'
                for ii in range(self.ndims):
                    bw_white[ii, ii] = self._compute_bandwidth(
                        bw_method[ii], param=(ii, ii))[0]
                bw_method = 'diagonal'
            elif np.shape(bw_method) == (ndims, ndims):
                for ii, jj in np.ndindex(ndims, ndims):
                    bw_white[ii, jj] = self._compute_bandwidth(
                        bw_method[ii, jj], param=(ii, jj))[0]
                bw_method = 'matrix'
            else:
                raise ValueError("`bw_method` have shape (1,), (N,) or (N,) for `N` dimensions!")

        bw_cov = self._data_cov * (bw_white ** 2)
        try:
            bw_cov_inv = np.linalg.inv(bw_cov)
        except np.linalg.LinAlgError:
            logging.warning("WARNING: singular `bw_cov` matrix, trying SVD...")
            bw_cov_inv = np.linalg.pinv(bw_cov)

        self.bw_white = bw_white
        self.bw_method = bw_method
        self._bw_method = _bw_method
        self.bw_cov = bw_cov
        self.bw_cov_inv = bw_cov_inv
        self.bw_norm = np.sqrt(np.linalg.det(2*np.pi*self.bw_cov))
        return

    def _compute_bandwidth(self, bw_method, param=None):
        if bw_method is None:
            bw_method = self.bw_method_default

        if isinstance(bw_method, six.string_types):
            if bw_method == 'scott':
                bandwidth = self.scott_factor(param=param)
            elif bw_method == 'silverman':
                bandwidth = self.silverman_factor(param=param)
            else:
                msg = "Unrecognized bandwidth str specification '{}'!".format(bw_method)
                raise ValueError(msg)

        elif np.isscalar(bw_method):
            if np.isclose(bw_method, 0.0):
                msg = "`bw_method` '{}' for param '{}' cannot be zero!".format(bw_method, param)
                raise ValueError(msg)

            bandwidth = bw_method
            bw_method = 'constant scalar'

        elif callable(bw_method):
            bw_method = bw_method
            bandwidth = bw_method(self, param=param)

        else:
            raise ValueError("Unrecognized `bw_method` '{}'!".format(bw_method))

        return bandwidth, bw_method

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using bandwidth_func().
        """

        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_cov'):
            cov = np.cov(self.dataset, rowvar=True, bias=False, aweights=self.weights)
            self._data_cov = np.atleast_2d(cov)
        # if not hasattr(self, '_data_inv_cov'):
        #     self._data_inv_cov = np.linalg.inv(self._data_cov)

        return

    @property
    def weights(self):
        try:
            if self._weights is None:
                raise AttributeError
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.nvals)/self.nvals
            return self._weights

    @property
    def neff(self):
        try:
            if self._neff is None:
                raise AttributeError
            return self._neff
        except AttributeError:
            self._neff = 1.0 / np.sum(self.weights**2)
            return self._neff
