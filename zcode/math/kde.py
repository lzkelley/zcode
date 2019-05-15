"""
"""

import six
import logging

import scipy as sp
import scipy.special  # noqa

import numpy as np

import zcode.math as zmath

__all__ = ['gaussian_kde']


class gaussian_kde(object):
    """
    """
    def __init__(self, dataset, bw_scale=None, bw_method=None, weights=None, neff=None, warn=True):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.ndims, self.nvals = self.dataset.shape
        if bw_scale is None:
            bw_scale = 1.0

        self.bw_scale = bw_scale
        if warn and (not np.isclose(bw_scale, 1.0)):
            logging.warning("WARNING: rescaling the bandwidth by `{:.4e}`".format(bw_scale))

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

            if neff is None:
                neff = 1/np.sum(self._weights**2)

        self._neff = neff
        self.set_bandwidth(bw_method=bw_method)
        return

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
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
        whitening = sp.linalg.cholesky(self.inv_cov)
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

        result = result / self._norm_factor

        return result

    def integrate_kde(self, other):
        """
        Computes the integral of the product of this  kernel density estimate
        with another.

        Parameters
        ----------
        other : gaussian_kde instance
            The other kde.

        Returns
        -------
        value : scalar
            The result of the integral.

        Raises
        ------
        ValueError
            If the KDEs have different dimensionality.

        """
        if other.d != self.ndims:
            raise ValueError("KDEs are not the same dimensionality")

        # we want to iterate over the smallest number of points
        if other.n < self.nvals:
            small = other
            large = self
        else:
            small = self
            large = other

        sum_cov = small.covariance + large.covariance
        sum_cov_chol = np.linalg.cho_factor(sum_cov)
        result = 0.0
        for i in range(small.n):
            mean = small.dataset[:, i, np.newaxis]
            diff = large.dataset - mean
            tdiff = np.linalg.cho_solve(sum_cov_chol, diff)

            energies = np.sum(diff * tdiff, axis=0) / 2.0
            result += np.sum(np.exp(-energies)*large.weights, axis=0)*small.weights[i]

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = np.power(2 * np.pi, sum_cov.shape[0] / 2.0) * sqrt_det

        result /= norm_const

        return result

    def resample(self, size=None):
        if size is None:
            size = int(self.neff)

        # Draw from the smoothing kernel
        # here the `cov` includes the bandwidth
        try:
            norm = np.random.multivariate_normal(np.zeros(self.ndims), self.cov, size=size)
        except ValueError as err:
            logging.error("Failed to construct `multivariate_normal`!  " + str(err))
            logging.error("cov = '{}'".format(self.cov))
            logging.error("dataset = '{}'".format(zmath.stats_str(self.dataset)))
            raise

        norm = np.transpose(norm)
        # Draw randomly from the given data points, proportionally to their weights
        indices = np.random.choice(self.nvals, size=size, p=self.weights)
        means = self.dataset[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm

        return samps

    def scotts_factor(self):
        """Compute Scott's factor.

        Returns
        -------
        s : float
            Scott's factor.
        """
        return np.power(self.neff, -1./(self.ndims+4))

    def silverman_factor(self):
        """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
        return np.power(self.neff*(self.ndims+2.0)/4.0, -1./(self.ndims+4))

    def set_bandwidth(self, bw_method=None):
        if bw_method is None:
            self.bandwidth_func = self.scotts_factor
        elif bw_method == 'scott':
            self.bandwidth_func = self.scotts_factor
        elif bw_method == 'silverman':
            self.bandwidth_func = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, six.string_types):
            self._bw_method = 'use constant'
            self.bandwidth_func = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.bandwidth_func = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()
        return

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using bandwidth_func().
        """

        # This is the bandwidth for whitened data (i.e. sigma_i = 1 for all i)
        #  sqrt(H)
        bandwidth_white = self.bandwidth_func()
        # print("{:.4e} {:.4e}".format(bandwidth_white, self.dataset.size))
        bandwidth_white *= self.bw_scale
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_cov'):
            cov = np.cov(self.dataset, rowvar=True, bias=False, aweights=self.weights)
            self._data_cov = np.atleast_2d(cov)
        if not hasattr(self, '_data_inv_cov'):
            self._data_inv_cov = np.linalg.inv(self._data_cov)

        HH = bandwidth_white**2
        # This is the smoothing-kernel matrix H_ij, for the full data
        self.cov = self._data_cov * HH
        self.inv_cov = self._data_inv_cov / HH
        # Including `2pi` in the determinant takes it to the d'th power.
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.cov))
        self.bandwidth_white = bandwidth_white
        return

    def pdf(self, x):
        """
        Evaluate the estimated pdf on a provided set of points.

        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
        docstring for more details.

        """
        return self.evaluate(x)

    def logpdf(self, x):
        """
        Evaluate the log of the estimated pdf on a provided set of points.
        """

        points = np.atleast_2d(x)

        d, m = points.shape
        if d != self.ndims:
            if d == 1 and m == self.ndims:
                # points was passed in as a row vector
                points = np.reshape(points, (self.ndims, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.ndims)
                raise ValueError(msg)

        if m >= self.nvals:
            # there are more points than data, so loop over data
            energy = np.zeros((self.nvals, m), dtype=float)
            for i in range(self.nvals):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy[i] = np.sum(diff*tdiff, axis=0) / 2.0
            result = sp.special.logsumexp(-energy.T, b=self.weights / self._norm_factor, axis=1)
        else:
            # loop over points
            result = np.zeros((m,), dtype=float)
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = sp.special.logsumexp(-energy, b=self.weights / self._norm_factor)

        return result

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.nvals)/self.nvals
            return self._weights

    @property
    def neff(self):
        neff = self._neff
        if neff is None:
            neff = 1/np.sum(self.weights**2)
            self._neff = neff

        return neff
