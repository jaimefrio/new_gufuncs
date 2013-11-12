from __future__ import division, absolute_import, print_function
import numpy as np

from ._new_gu_kernels import _interp, _bincount, _bincount_wt, _minmax

def interp(x, xp, fp, left=None, right=None):
    left = fp[..., 0] if left is None else left
    right = fp[..., -1] if right is None else right
    return _interp(x, xp, fp, left, right)

def bincount(x, weights=None, minlength=None):
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
    