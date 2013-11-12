from __future__ import division, absolute_import, print_function
import numpy as np

from ._new_gu_kernels import _interp, _bincount, _bincount_wt, _minmax

def interp(x, xp, fp, left=None, right=None):
    """
    One-dimensional linear interpolation on the last dimension, broadcast
    on the rest.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given values at discrete data-points.

    Parameters
    ----------
    x : (..., n) array_like
        The x-coordinates of the interpolated values.
    xp : (..., p) array_like
        The x-coordinates of the data points, must be increasing along last
        dimension.
    fp : (..., p) array_like
        The y-coordinates of the data points, last dimensions must be
        broadcastable to `xp`.
    left : (...) float, optional
        Values to return for `x < xp[0]`, default is `fp[..., 0]`.
    right : (...) float, optional
        Values to return for `x > xp[-1]`, default is `fp[..., -1]`.
        
    Returns
    -------
    y : (..., n) ndarray
        The interpolated values, same shape as `x`.
    """
    left = fp[..., 0] if left is None else left
    right = fp[..., -1] if right is None else right
    return _interp(x, xp, fp, left, right)

def bincount(x, weights=None, minlength=None):
    """
    Count number of occurrences of each value in array of non-negative ints
    on the last dimension, broadcast on the rest.

    The number of bins (of size 1) is one larger than the largest value in `x`.
    If `minlength` is specified, there will be at least this number of bins in
    the output array (though it will be longer if necessary, depending on the
    contents of `x`). Each bin gives the number of occurrences of its index
    value in `x`. If `weights` is specified the input array is weighted by it,
    i.e. if a value `n` is found at position `i`, `out[n] += weight[i]`
    instead of `out[n] += 1`.

    Parameters
    ----------	
    x : (..., m) array_like, nonnegative ints
        Input array.
    weights : (..., m) array_like, optional
        Weights, array of the same shape as x.
    minlength : (...) int, optional
        New in version 1.6.0.
        A minimum number of bins for the output array.
        
    Returns
    -------
    out : (..., n) ndarray
        The result of binning the input array. The length of `out` is equal
        to `np.amax(x)+1`.
    """
    out_type = {'?': np.intp,
                'b' : np.intp, 'B' : np.uintp,
                'h' : np.intp, 'H' : np.uintp,
                'l' : np.intp, 'L' : np.uintp,
                'q' : np.longlong, 'Q' : np.ulonglong,
                'f' : np.float, 'F' : np.cfloat,
                'd' : np.double, 'D' : np.cdouble,
                'g' : np.longdouble, 'G' : np.clongdouble,
               }
    x = np.asarray(x)
    min_, max_ = _minmax(x.ravel())
    if min_ < 0:
        msg = 'The first argument of bincount must be non-negative'
        raise ValueError(msg)
    n = max_ + 1 if minlength is None or minlength <= max_ else minlength
    if weights is None:
        out = np.zeros(x.shape[:-1] + (n,), dtype=np.intp)
        _bincount(x, out=out)
    else:
        weights = np.asarray(weights)
        shape = np.broadcast(x, weights).shape
        out = np.zeros(shape[:-1] + (n,), dtype=out_type[weights.dtype.char])
        _bincount_wt(x, weights, out=out)
    return out
    