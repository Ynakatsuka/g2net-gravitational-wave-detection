import json
import random
import string
import warnings
from warnings import warn

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import linalg


def rgb2xyz(rgb):
    """RGB to XYZ color space conversion.
    Parameters
    ----------
    rgb : (..., 3) array_like
        The image in RGB format. Final dimension denotes channels.
    Returns
    -------
    out : (..., 3) ndarray
        The image in XYZ format. Same dimensions as input.
    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3).
    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts from sRGB.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _prepare_colorarray(rgb).copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr @ xyz_from_rgb.T.astype(arr.dtype)


def lab2xyz(lab, illuminant="D65", observer="2"):
    """CIE-LAB to XYZcolor space conversion.
    Parameters
    ----------
    lab : array_like
        The image in lab format, in a 3-D array of shape ``(.., .., 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in XYZ format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    UserWarning
        If any of the pixels are invalid (Z < 0).
    Notes
    -----
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values x_ref
    = 95.047, y_ref = 100., z_ref = 108.883. See function 'get_xyz_coords' for
    a list of supported illuminants.
    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] https://en.wikipedia.org/wiki/Lab_color_space
    """

    arr = _prepare_colorarray(lab).copy()

    L, a, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    if np.any(z < 0):
        invalid = np.nonzero(z < 0)
        warn(
            "Color data out of range: Z < 0 in %s pixels" % invalid[0].size,
            stacklevel=2,
        )
        z[invalid] = 0

    out = np.dstack([x, y, z])

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    out *= xyz_ref_white
    return out


def xyz2lab(xyz, illuminant="D65", observer="2"):
    """XYZ to CIE-LAB color space conversion.
    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in CIE-LAB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape ``(.., ..,[ ..,] 3)``.
    ValueError
        If either the illuminant or the observer angle is unsupported or
        unknown.
    Notes
    -----
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] https://en.wikipedia.org/wiki/Lab_color_space
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2lab
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_lab = xyz2lab(img_xyz)
    """
    arr = _prepare_colorarray(xyz)

    xyz_ref_white = get_xyz_coords(illuminant, observer)

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = np.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16.0 / 116.0

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)


def lab2rgb(lab, illuminant="D65", observer="2"):
    """Lab to RGB color space conversion.
    Parameters
    ----------
    lab : array_like
        The image in Lab format, in a 3-D array of shape ``(.., .., 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    Notes
    -----
    This function uses lab2xyz and xyz2rgb.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2rgb(lab2xyz(lab, illuminant, observer))


def rgb2lab(rgb, illuminant="D65", observer="2"):
    """RGB to lab color space conversion.
    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in Lab format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    Raises
    ------
    ValueError
        If `rgb` is not a 3- or 4-D array of shape ``(.., ..,[ ..,] 3)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    Notes
    -----
    This function uses rgb2xyz and xyz2lab.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)


def lch2lab(lch):
    """CIE-LCH to CIE-LAB color space conversion.
    LCH is the cylindrical representation of the LAB (Cartesian) colorspace
    Parameters
    ----------
    lch : array_like
        The N-D image in CIE-LCH format. The last (``N+1``-th) dimension must
        have at least 3 elements, corresponding to the ``L``, ``a``, and ``b``
        color channels.  Subsequent elements are copied.
    Returns
    -------
    out : ndarray
        The image in LAB format, with same shape as input `lch`.
    Raises
    ------
    ValueError
        If `lch` does not have at least 3 color channels (i.e. l, c, h).
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2lab, lch2lab
    >>> img = data.astronaut()
    >>> img_lab = rgb2lab(img)
    >>> img_lch = lab2lch(img_lab)
    >>> img_lab2 = lch2lab(img_lch)
    """
    lch = _prepare_lab_array(lch)

    c, h = lch[..., 1], lch[..., 2]
    lch[..., 1], lch[..., 2] = c * np.cos(h), c * np.sin(h)
    return lch


def _prepare_lab_array(arr):
    """Ensure input for lab2lch, lch2lab are well-posed.
    Arrays must be in floating point and have at least 3 elements in
    last dimension.  Return a new array.
    """
    arr = np.asarray(arr)
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError("Input array has less than 3 color channels")
    return img_as_float(arr, force_copy=True)


def get_xyz_coords(illuminant, observer):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.
    Parameters
    ----------
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    (x, y, z) : tuple
        A tuple with 3 elements containing the XYZ coordinates of the given
        illuminant.
    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    illuminant = illuminant.upper()
    try:
        return illuminants[illuminant][observer]
    except KeyError:
        raise ValueError(
            "Unknown illuminant/observer combination\
        ('{0}', '{1}')".format(
                illuminant, observer
            )
        )


def _prepare_colorarray(arr):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.ndim not in [3, 4] or arr.shape[-1] != 3:
        msg = (
            "the input array must be have a shape == (.., ..,[ ..,] 3)), "
            + "got ("
            + (", ".join(map(str, arr.shape)))
            + ")"
        )
        raise ValueError(msg)

    return img_as_float(arr)


def xyz2rgb(xyz):
    """XYZ to RGB color space conversion.
    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape ``(.., .., 3)``.
    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts to sRGB.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2rgb
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_rgb = xyz2rgb(img_xyz)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr


def _convert(matrix, arr):
    """Do the color space conversion.
    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.
    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """
    arr = _prepare_colorarray(arr)

    return arr @ matrix.T.copy()


# ---------------------------------------------------------------
# Primaries for the coordinate systems
# ---------------------------------------------------------------
cie_primaries = np.array([700, 546.1, 435.8])
sb_primaries = np.array([1.0 / 155, 1.0 / 190, 1.0 / 225]) * 1e5

# ---------------------------------------------------------------
# Matrices that define conversion between different color spaces
# ---------------------------------------------------------------

# From sRGB specification
xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

rgb_from_xyz = linalg.inv(xyz_from_rgb)

# From https://en.wikipedia.org/wiki/CIE_1931_color_space
# Note: Travis's code did not have the divide by 0.17697
xyz_from_rgbcie = (
    np.array(
        [[0.49, 0.31, 0.20], [0.17697, 0.81240, 0.01063], [0.00, 0.01, 0.99]]
    )
    / 0.17697
)

rgbcie_from_xyz = linalg.inv(xyz_from_rgbcie)

# construct matrices to and from rgb:
rgbcie_from_rgb = rgbcie_from_xyz @ xyz_from_rgb
rgb_from_rgbcie = rgb_from_xyz @ xyz_from_rgbcie


gray_from_rgb = np.array([[0.2125, 0.7154, 0.0721], [0, 0, 0], [0, 0, 0]])

yuv_from_rgb = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026],
    ]
)

rgb_from_yuv = linalg.inv(yuv_from_rgb)

yiq_from_rgb = np.array(
    [
        [0.299, 0.587, 0.114],
        [0.59590059, -0.27455667, -0.32134392],
        [0.21153661, -0.52273617, 0.31119955],
    ]
)

rgb_from_yiq = linalg.inv(yiq_from_rgb)

ypbpr_from_rgb = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)

rgb_from_ypbpr = linalg.inv(ypbpr_from_rgb)

ycbcr_from_rgb = np.array(
    [
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214],
    ]
)

rgb_from_ycbcr = linalg.inv(ycbcr_from_rgb)

ydbdr_from_rgb = np.array(
    [[0.299, 0.587, 0.114], [-0.45, -0.883, 1.333], [-1.333, 1.116, 0.217]]
)

rgb_from_ydbdr = linalg.inv(ydbdr_from_rgb)


# CIE LAB constants for Observer=2A, Illuminant=D65
# NOTE: this is actually the XYZ values for the illuminant above.
lab_ref_white = np.array([0.95047, 1.0, 1.08883])


# XYZ coordinates of the illuminants, scaled to [0, 1]. For each illuminant I
# we have:
#
#   illuminant[I][0] corresponds to the XYZ coordinates for the 2 degree
#   field of view.
#
#   illuminant[I][1] corresponds to the XYZ coordinates for the 10 degree
#   field of view.
#
# The XYZ coordinates are calculated from [1], using the formula:
#
#   X = x * ( Y / y )
#   Y = Y
#   Z = ( 1 - x - y ) * ( Y / y )
#
# where Y = 1. The only exception is the illuminant "D65" with aperture angle
# 2, whose coordinates are copied from 'lab_ref_white' for
# backward-compatibility reasons.
#
#     References
#    ----------
#    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant

illuminants = {
    "A": {
        "2": (1.098466069456375, 1, 0.3558228003436005),
        "10": (1.111420406956693, 1, 0.3519978321919493),
    },
    "D50": {
        "2": (0.9642119944211994, 1, 0.8251882845188288),
        "10": (0.9672062750333777, 1, 0.8142801513128616),
    },
    "D55": {
        "2": (0.956797052643698, 1, 0.9214805860173273),
        "10": (0.9579665682254781, 1, 0.9092525159847462),
    },
    "D65": {
        "2": (0.95047, 1.0, 1.08883),  # This was: `lab_ref_white`
        "10": (0.94809667673716, 1, 1.0730513595166162),
    },
    "D75": {
        "2": (0.9497220898840717, 1, 1.226393520724154),
        "10": (0.9441713925645873, 1, 1.2064272211720228),
    },
    "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
}


# For integers Numpy uses `_integer_types` basis internally, and builds a leaky
# `np.XintYY` abstraction on top of it. This leads to situations when, for
# example, there are two np.Xint64 dtypes with the same attributes but
# different object references. In order to avoid any potential issues,
# we use the basis dtypes here. For more information, see:
# - https://github.com/scikit-image/scikit-image/issues/3043
# For convenience, for these dtypes we indicate also the possible bit depths
# (some of them are platform specific). For the details, see:
# http://www.unix.org/whitepapers/64bit.html
_integer_types = (
    np.byte,
    np.ubyte,  # 8 bits
    np.short,
    np.ushort,  # 16 bits
    np.intc,
    np.uintc,  # 16 or 32 or 64 bits
    np.int_,
    np.uint,  # 32 or 64 bits
    np.longlong,
    np.ulonglong,
)  # 64 bits
_integer_ranges = {
    t: (np.iinfo(t).min, np.iinfo(t).max) for t in _integer_types
}
dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}
dtype_range.update(_integer_ranges)

_supported_types = list(dtype_range.keys())


def dtype_limits(image, clip_negative=False):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.
    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
    Returns
    -------
    imin, imax : tuple
        Lower and upper intensity limits.
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def _dtype_itemsize(itemsize, *dtypes):
    """Return first of `dtypes` with itemsize greater than `itemsize`
    Parameters
    ----------
    itemsize: int
        The data type object element size.
    Other Parameters
    ----------------
    *dtypes:
        Any Object accepted by `np.dtype` to be converted to a data
        type object
    Returns
    -------
    dtype: data type object
        First of `dtypes` with itemsize greater than `itemsize`.
    """
    return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)


def _dtype_bits(kind, bits, itemsize=1):
    """Return dtype of `kind` that can store a `bits` wide unsigned int
    Parameters:
    kind: str
        Data type kind.
    bits: int
        Desired number of bits.
    itemsize: int
        The data type object element size.
    Returns
    -------
    dtype: data type object
        Data type of `kind` that can store a `bits` wide unsigned int
    """

    s = next(
        i
        for i in (itemsize,) + (2, 4, 8)
        if bits < (i * 8) or (bits == (i * 8) and kind == "u")
    )

    return np.dtype(kind + str(s))


def _scale(a, n, m, copy=True):
    """Scale an array of unsigned/positive integers from `n` to `m` bits.
    Numbers can be represented exactly only if `m` is a multiple of `n`.
    Parameters
    ----------
    a : ndarray
        Input image array.
    n : int
        Number of bits currently used to encode the values in `a`.
    m : int
        Desired number of bits to encode the values in `out`.
    copy : bool, optional
        If True, allocates and returns new array. Otherwise, modifies
        `a` in place.
    Returns
    -------
    out : array
        Output image array. Has the same kind as `a`.
    """
    kind = a.dtype.kind
    if n > m and a.max() < 2 ** m:
        mnew = int(np.ceil(m / 2) * 2)
        if mnew > m:
            dtype = "int{}".format(mnew)
        else:
            dtype = "uint{}".format(mnew)
        n = int(np.ceil(n / 2) * 2)
        warn(
            "Downcasting {} to {} without scaling because max "
            "value {} fits in {}".format(a.dtype, dtype, a.max(), dtype),
            stacklevel=3,
        )
        return a.astype(_dtype_bits(kind, m))
    elif n == m:
        return a.copy() if copy else a
    elif n > m:
        # downscale with precision loss
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.floor_divide(
                a, 2 ** (n - m), out=b, dtype=a.dtype, casting="unsafe"
            )
            return b
        else:
            a //= 2 ** (n - m)
            return a
    elif m % n == 0:
        # exact upscale to a multiple of `n` bits
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.multiply(a, (2 ** m - 1) // (2 ** n - 1), out=b, dtype=b.dtype)
            return b
        else:
            a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
            a *= (2 ** m - 1) // (2 ** n - 1)
            return a
    else:
        # upscale to a multiple of `n` bits,
        # then downscale with precision loss
        o = (m // n + 1) * n
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, o))
            np.multiply(a, (2 ** o - 1) // (2 ** n - 1), out=b, dtype=b.dtype)
            b //= 2 ** (o - m)
            return b
        else:
            a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
            a *= (2 ** o - 1) // (2 ** n - 1)
            a //= 2 ** (o - m)
            return a


def convert(image, dtype, force_copy=False, uniform=False):
    """
    Convert an image to the requested data-type.
    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).
    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.
    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.
    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.
    .. versionchanged :: 0.15
        ``convert`` no longer warns about possible precision or sign
        information loss. See discussions on these warnings at:
        https://github.com/scikit-image/scikit-image/issues/2602
        https://github.com/scikit-image/scikit-image/issues/543#issuecomment-208202228
        https://github.com/scikit-image/scikit-image/pull/3575
    References
    ----------
    .. [1] DirectX data conversion rules.
           https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.
    """
    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype("float64")
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `np.floating` will allow
    #   `float32` and `float64` arrays through)

    if np.issubdtype(dtype_in, np.obj2sctype(dtype)):
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(
            "Can not convert from {} to {}.".format(dtypeobj_in, dtypeobj_out)
        )

    if kind_in in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in "ui":
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == "b":
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == "b":
        result = image.astype(dtype_out)
        if kind_out != "f":
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == "f":
        if kind_out == "f":
            # float -> float
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, np.float32, np.float64
        )

        if not uniform:
            if kind_out == "u":
                image_out = np.multiply(
                    image, imax_out, dtype=computation_type
                )
            else:
                image_out = np.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type
                )
                image_out -= 1.0 / 2.0
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == "u":
            image_out = np.multiply(
                image, imax_out + 1, dtype=computation_type
            )
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(
                image,
                (imax_out - imin_out + 1.0) / 2.0,
                dtype=computation_type,
            )
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == "f":
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, np.float32, np.float64
        )

        if kind_in == "u":
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == "u":
        if kind_out == "i":
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == "u":
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting="unsafe")
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits("i", itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)


def img_as_float32(image, force_copy=False):
    """Convert an image to single-precision (32-bit) floating point format.
    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    Returns
    -------
    out : ndarray of float32
        Output image.
    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].
    """
    return convert(image, np.float32, force_copy)


def img_as_float64(image, force_copy=False):
    """Convert an image to double-precision (64-bit) floating point format.
    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    Returns
    -------
    out : ndarray of float64
        Output image.
    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].
    """
    return convert(image, np.float64, force_copy)


def img_as_float(image, force_copy=False):
    """Convert an image to floating point format.
    This function is similar to `img_as_float64`, but will not convert
    lower-precision floating point arrays to `float64`.
    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    Returns
    -------
    out : ndarray of float
        Output image.
    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].
    """
    return convert(image, np.floating, force_copy)


def lch2rgb(x):
    return lab2rgb(lch2lab([[x]]))[0][0]


colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, l))
for l in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, l))
red_transparent_blue = LinearSegmentedColormap.from_list(
    "red_transparent_blue", colors
)


blue_lch = [54.0, 70.0, 4.6588]
red_lch = [54.0, 90.0, 0.35470565 + 2 * np.pi]
gray_lch = [55.0, 0.0, 0.0]
blue_rgb = lch2rgb(blue_lch)
red_rgb = lch2rgb(red_lch)


# From: https://groups.google.com/forum/m/#!topic/openrefine/G7_PSdUeno0
def ordinal_str(n):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(
        4 if 10 <= n % 100 < 20 else n % 10, "th"
    )


# TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
def text(
    shap_values,
    num_starting_labels=0,
    group_threshold=1,
    separator="",
    xmin=None,
    xmax=None,
    cmax=None,
):
    """Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shap values for a string(# input_tokens x output_tokens).

    num_starting_labels : int
        Number of tokens (sorted in decending order by corresponding SHAP values) that are uncovered in the initial view. When set to 0 all tokens
        covered.

    group_threshold : float
        The threshold used to group tokens based on interaction affects of SHAP values.

    separator : string
        The string seperator that joins tokens grouped by interation effects and unbroken string spans.

    xmin : float
        Minimum shap value bound.

    xmax : float
        Maximum shap value bound.

    cmax : float
        Maximum absolute shap value for sample. Used for scaling colors for input tokens.

    """
    from IPython.core.display import HTML, display

    def values_min_max(values, base_values):
        """Used to pick our axis limits."""
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2 and (
        shap_values.output_names is None
        or isinstance(shap_values.output_names, str)
    ):
        xmin = 0
        xmax = 0
        cmax = 0

        for i in range(0, len(shap_values)):

            values, clustering = unpack_shap_explanation_contents(
                shap_values[i]
            )
            tokens, values, group_sizes = process_shap_values(
                shap_values[i].data,
                values,
                group_threshold,
                separator,
                clustering,
            )

            if i == 0:
                xmin, xmax, cmax = values_min_max(
                    values, shap_values[i].base_values
                )
                continue

            xmin_i, xmax_i, cmax_i = values_min_max(
                values, shap_values[i].base_values
            )
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i

        for i in range(len(shap_values)):
            display(HTML("<br/><b>" + ordinal_str(i) + " instance:</b><br/>"))
            text(
                shap_values[i],
                num_starting_labels=num_starting_labels,
                group_threshold=group_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
            )
        return

    elif len(shap_values.shape) == 2 and shap_values.output_names is not None:
        text_to_text(shap_values)
        return
    elif len(shap_values.shape) == 3:
        for i in range(len(shap_values)):
            display(HTML("<br/><b>" + ordinal_str(i) + " instance:</b><br/>"))
            text(shap_values[i])
        return

    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(
        shap_values.values, shap_values.base_values
    )
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new

    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(
        shap_values.data, values, group_threshold, separator, clustering
    )

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    maxv = values.max()
    minv = values.min()
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [
        t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "")
        for t in tokens
    ]
    out += svg_force_plot(
        values,
        shap_values.base_values,
        shap_values.base_values + values.sum(),
        encoded_tokens,
        uuid,
        xmin,
        xmax,
    )

    for i in range(len(tokens)):
        scaled_value = 0.5 + 0.5 * values[i] / cmax
        color = red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"

        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])

        # the HTML for this token
        out += (
            "<div style='display: "
            + wrapper_display
            + "; text-align: center;'>"
            + "<div style='display: "
            + label_display
            + "; color: #999; padding-top: 0px; font-size: 12px;'>"
            + value_label
            + "</div>"
            + f"<div id='_tp_{uuid}_ind_{i}'"
            + "style='display: inline; background: rgba"
            + str(color)
            + "; border-radius: 3px; padding: 0px'"
            + "onclick=\"if (this.previousSibling.style.display == 'none') {"
            + "this.previousSibling.style.display = 'block';"
            + "this.parentNode.style.display = 'inline-block';"
            + "} else {"
            + "this.previousSibling.style.display = 'none';"
            + "this.parentNode.style.display = 'inline';"
            + "}"
            + '"'
            + f"onmouseover=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;"
            + '"'
            + f"onmouseout=\"document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;"
            + '"'
            + ">"
            + tokens[i]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            + "</div>"
            + "</div>"
        )

    display(HTML(out))
    return out


def process_shap_values(
    tokens,
    values,
    group_threshold,
    separator,
    clustering=None,
    return_meta_data=False,
):

    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:

        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                + "tokens if shap_values.clustering is None! When passing hierarchical "
                + "attributions the clustering is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i, 0])
            ri = int(clustering[i, 1])
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = (
                lower_values[li] + lower_values[ri] + values[M + i]
            )
            max_values[i + M] = max(
                abs(values[M + i]) / len(groups[M + i]),
                max_values[li],
                max_values[ri],
            )

        # compute the upper_values
        upper_values = np.zeros(len(values))

        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i - M, 0])
            ri = int(clustering[i - M, 1])
            upper_values[i] = value
            value += values[i]
            #             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
            #             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)

        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):

            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:

                # compute the dividend at internal nodes
                li = int(clustering[i - M, 0])
                ri = int(clustering[i - M, 1])
                dv = abs(values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if dv > group_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i

                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i

                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)

        merge_tokens(
            new_tokens, new_values, group_sizes, len(group_values) - 1
        )

        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens)
    else:
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)

    if return_meta_data:
        return (
            tokens,
            values,
            group_sizes,
            token_id_to_node_id_mapping,
            collapsed_node_ids,
        )
    else:
        return tokens, values, group_sizes


def svg_force_plot(values, base_values, fx, tokens, uuid, xmin, xmax):
    def xpos(xval):
        return 100 * (xval - xmin) / (xmax - xmin)

    s = ""
    s += '<svg width="100%" height="80px">'

    ### x-axis marks ###

    # draw x axis line
    s += '<line x1="0" y1="33" x2="100%" y2="33" style="stroke:rgb(150,150,150);stroke-width:1" />'

    # draw base value
    def draw_tick_mark(xval, label=None, bold=False):
        s = ""
        s += (
            '<line x1="%f%%" y1="33" x2="%f%%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" />'
            % ((xpos(xval),) * 2)
        )
        if not bold:
            s += (
                '<text x="%f%%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">%f</text>'
                % (xpos(xval), xval)
            )
        else:
            s += (
                '<text x="%f%%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" font-weight="bold" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">%f</text>'
                % (xpos(xval), xval)
            )
            s += (
                '<text x="%f%%" y="27" font-size="13px" font-weight="bold" fill="rgb(0,0,0)" dominant-baseline="bottom" text-anchor="middle">%f</text>'
                % (xpos(xval), xval)
            )
        if label is not None:
            s += (
                '<text x="%f%%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle">%s</text>'
                % (xpos(xval), label)
            )
        return s

    s += draw_tick_mark(base_values, label="base value")
    tick_interval = (xmax - xmin) / 7
    side_buffer = (xmax - xmin) / 14
    for i in range(1, 10):
        pos = base_values - i * tick_interval
        if pos < xmin + side_buffer:
            break
        s += draw_tick_mark(pos)
    for i in range(1, 10):
        pos = base_values + i * tick_interval
        if pos > xmax - side_buffer:
            break
        s += draw_tick_mark(pos)
    s += draw_tick_mark(fx, bold=True, label="f(x)")

    ### Positive value marks ###

    red = tuple(red_rgb * 255)
    light_red = (255, 195, 213)

    # draw base red bar
    x = fx - values[values > 0].sum()
    w = 100 * values[values > 0].sum() / (xmax - xmin)
    s += f'<rect x="{xpos(x)}%" width="{w}%" y="40" height="18" style="fill:rgb{red}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] > 0]
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # a line under the bar to animate
        s += f'<line x1="{xpos(pos)}%" x2="{xpos(last_pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{red};stroke-width:2; opacity: 0"/>'

        # the text label cropped and centered
        s += f'<text x="{(xpos(last_pos) + xpos(pos))/2}%" y="71" font-size="12px" id="_fs_{uuid}_ind_{ind}" fill="rgb{red}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'

        # the text label cropped and centered
        s += f'<svg x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%">'
        s += f'  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += f"  </svg>"
        s += f"</svg>"

        last_pos = pos

    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({2*j-8},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += f"  </svg>"
                s += f"</g>"

        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate({2*j-0},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += f"  </svg>"
                s += f"</g>"

        last_pos = pos

    # center padding
    s += f'<rect transform="translate(-8,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{red}"/>'

    # cover up a notch at the end of the red bar
    pos = fx - values[values > 0].sum()
    s += f'<g transform="translate(-11.5,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += f'    <path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += f"  </svg>"
    s += f"</g>"

    # draw the light red divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += f'<g transform="translate(-1.5,0)">'
            s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{light_red};stroke-width:2" />'
            s += f"  </svg>"
            s += f"</g>"

        # mouse over rectangle
        s += f'<rect x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%"'
        s += f'      onmouseover="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'underline';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f'"'
        s += f'      onmouseout="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'none';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f'" style="fill:rgb(0,0,0,0)" />'

        last_pos = pos

    ### Negative value marks ###

    blue = tuple(blue_rgb * 255)
    light_blue = (208, 230, 250)

    # draw base blue bar
    w = 100 * -values[values < 0].sum() / (xmax - xmin)
    s += f'<rect x="{xpos(fx)}%" width="{w}%" y="40" height="18" style="fill:rgb{blue}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] < 0]
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # a line under the bar to animate
        s += f'<line x1="{xpos(last_pos)}%" x2="{xpos(pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{blue};stroke-width:2; opacity: 0"/>'

        # the value text
        s += f'<text x="{(xpos(last_pos) + xpos(pos))/2}%" y="71" font-size="12px" fill="rgb{blue}" id="_fs_{uuid}_ind_{ind}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'

        # the text label cropped and centered
        s += f'<svg x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%">'
        s += f'  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle">{tokens[ind].strip()}</text>'
        s += f"  </svg>"
        s += f"</svg>"

        last_pos = pos

    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({-2*j+2},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += f"  </svg>"
                s += f"</g>"

        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate(-{2*j+8},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += f"  </svg>"
                s += f"</g>"

        last_pos = pos

    # center padding
    s += f'<rect transform="translate(0,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{blue}"/>'

    # cover up a notch at the end of the blue bar
    pos = fx - values[values < 0].sum()
    s += f'<g transform="translate(-6.0,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
    s += f'    <path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += f"  </svg>"
    s += f"</g>"

    # draw the light blue divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += f'<g transform="translate(-6.0,0)">'
            s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="visible" width="30">'
            s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{light_blue};stroke-width:2" />'
            s += f"  </svg>"
            s += f"</g>"

        # mouse over rectangle
        s += f'<rect x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%"'
        s += f'      onmouseover="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'underline';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 1;"
        s += f'"'
        s += f'      onmouseout="'
        s += f"document.getElementById('_tp_{uuid}_ind_{ind}').style.textDecoration = 'none';"
        s += f"document.getElementById('_fs_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f"document.getElementById('_fb_{uuid}_ind_{ind}').style.opacity = 0;"
        s += f'" style="fill:rgb(0,0,0,0)" />'

        last_pos = pos

    s += "</svg>"

    return s


def text_old(
    shap_values,
    tokens,
    partition_tree=None,
    num_starting_labels=0,
    group_threshold=1,
    separator="",
):
    """Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.
    """

    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(shap_values) != M:

        # make sure we were given a partition tree
        if partition_tree is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                + "tokens if partition_tree is None! When passing hierarchical "
                + "attributions the partition_tree is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(shap_values))
        lower_values[:M] = shap_values[:M]
        max_values = np.zeros(len(shap_values))
        max_values[:M] = np.abs(shap_values[:M])
        for i in range(partition_tree.shape[0]):
            li = partition_tree[i, 0]
            ri = partition_tree[i, 1]
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = (
                lower_values[li] + lower_values[ri] + shap_values[M + i]
            )
            max_values[i + M] = max(
                abs(shap_values[M + i]) / len(groups[M + i]),
                max_values[li],
                max_values[ri],
            )

        # compute the upper_values
        upper_values = np.zeros(len(shap_values))

        def lower_credit(upper_values, partition_tree, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = partition_tree[i - M, 0]
            ri = partition_tree[i - M, 1]
            upper_values[i] = value
            value += shap_values[i]
            #             lower_credit(upper_values, partition_tree, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
            #             lower_credit(upper_values, partition_tree, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, partition_tree, li, value * 0.5)
            lower_credit(upper_values, partition_tree, ri, value * 0.5)

        lower_credit(upper_values, partition_tree, len(shap_values) - 1)

        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_shap_values = []
        group_sizes = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):

            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)
            else:

                # compute the dividend at internal nodes
                li = partition_tree[i - M, 0]
                ri = partition_tree[i - M, 1]
                dv = abs(shap_values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if dv > group_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
                    new_values.append(group_values[i] / len(groups[i]))
                    group_sizes.append(len(groups[i]))
                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)

        merge_tokens(
            new_tokens, new_shap_values, group_sizes, len(group_values) - 1
        )

        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        shap_values = np.array(new_shap_values)
        group_sizes = np.array(group_sizes)
        M = len(tokens)
    else:
        group_sizes = np.ones(M)

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(shap_values))[:num_starting_labels]
    maxv = shap_values.max()
    minv = shap_values.min()
    out = ""
    for i in range(M):
        scaled_value = 0.5 + 0.5 * shap_values[i] / max(abs(maxv), abs(minv))
        color = red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"

        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(shap_values[i].round(3))
        else:
            value_label = (
                str((shap_values[i] * group_sizes[i]).round(3))
                + " / "
                + str(group_sizes[i])
            )

        # the HTML for this token
        out += (
            "<div style='display: "
            + wrapper_display
            + "; text-align: center;'>"
            + "<div style='display: "
            + label_display
            + "; color: #999; padding-top: 0px; font-size: 12px;'>"
            + value_label
            + "</div>"
            + "<div "
            + "style='display: inline; background: rgba"
            + str(color)
            + "; border-radius: 3px; padding: 0px'"
            + "onclick=\"if (this.previousSibling.style.display == 'none') {"
            + "this.previousSibling.style.display = 'block';"
            + "this.parentNode.style.display = 'inline-block';"
            + "} else {"
            + "this.previousSibling.style.display = 'none';"
            + "this.parentNode.style.display = 'inline';"
            + "}"
            + '"'
            + ">"
            + tokens[i]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            + "</div>"
            + "</div>"
        )

    from IPython.core.display import HTML, display

    return display(HTML(out))


def text_to_text(shap_values):

    from IPython.core.display import HTML, display

    # unique ID added to HTML elements and function to avoid collision of differnent instances
    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    saliency_plot_markup = saliency_plot(shap_values)
    heatmap_markup = heatmap(shap_values)

    html = f"""
    <html>
    <div id="{uuid}_viz_container">
      <div id="{uuid}_viz_header" style="padding:15px;border-style:solid;margin:5px;font-family:sans-serif;font-weight:bold;">
        Visualization Type:
        <select name="viz_type" id="{uuid}_viz_type" onchange="selectVizType_{uuid}(this)">
          <option value="heatmap" selected="selected">Input/Output - Heatmap</option>
          <option value="saliency-plot">Saliency Plot</option>
        </select>
      </div>
      <div id="{uuid}_content" style="padding:15px;border-style:solid;margin:5px;">
          <div id = "{uuid}_saliency_plot_container" class="{uuid}_viz_container" style="display:none">
              {saliency_plot_markup}
          </div>

          <div id = "{uuid}_heatmap_container" class="{uuid}_viz_container">
              {heatmap_markup}
          </div>
      </div>
    </div>
    </html>
    """

    javascript = f"""
    <script>
        function selectVizType_{uuid}(selectObject) {{

          /* Hide all viz */

            var elements = document.getElementsByClassName("{uuid}_viz_container")
          for (var i = 0; i < elements.length; i++){{
              elements[i].style.display = 'none';
          }}

          var value = selectObject.value;
          if ( value === "saliency-plot" ){{
              document.getElementById('{uuid}_saliency_plot_container').style.display  = "block";
          }}
          else if ( value === "heatmap" ) {{
              document.getElementById('{uuid}_heatmap_container').style.display  = "block";
          }}
        }}
    </script>
    """

    display(HTML(javascript + html))


def saliency_plot(shap_values):

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    unpacked_values, clustering = unpack_shap_explanation_contents(shap_values)
    (
        tokens,
        values,
        group_sizes,
        token_id_to_node_id_mapping,
        collapsed_node_ids,
    ) = process_shap_values(
        shap_values.data, unpacked_values[:, 0], 1, "", clustering, True
    )

    def compress_shap_matrix(shap_matrix, group_sizes):
        compressed_matrix = np.zeros(
            (group_sizes.shape[0], shap_matrix.shape[1])
        )
        counter = 0
        for index in range(len(group_sizes)):
            compressed_matrix[index, :] = np.sum(
                shap_matrix[counter : counter + group_sizes[index], :], axis=0
            )
            counter += group_sizes[index]

        return compressed_matrix

    compressed_shap_matrix = compress_shap_matrix(
        shap_values.values, group_sizes
    )

    # generate background colors of saliency plot

    def get_colors(shap_values):
        input_colors = []
        cmax = max(
            abs(compressed_shap_matrix.min()),
            abs(compressed_shap_matrix.max()),
        )
        for row_index in range(compressed_shap_matrix.shape[0]):
            input_colors_row = []
            for col_index in range(compressed_shap_matrix.shape[1]):
                scaled_value = (
                    0.5
                    + 0.5 * compressed_shap_matrix[row_index, col_index] / cmax
                )
                color = red_transparent_blue(scaled_value)
                color = "rgba" + str(
                    (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
                )
                input_colors_row.append(color)
            input_colors.append(input_colors_row)

        return input_colors

    model_output = shap_values.output_names

    input_colors = get_colors(shap_values)

    out = '<table border = "1" cellpadding = "5" cellspacing = "5" style="overflow-x:scroll;display:block;">'

    # add top row containing input tokens
    out += "<tr>"
    out += "<th></th>"

    for j in range(compressed_shap_matrix.shape[0]):
        out += (
            "<th>"
            + tokens[j]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            .replace("▁", "")
            .replace("Ġ", "")
            + "</th>"
        )
    out += "</tr>"

    for row_index in range(compressed_shap_matrix.shape[1]):
        out += "<tr>"
        out += (
            "<th>"
            + model_output[row_index]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            .replace("▁", "")
            .replace("Ġ", "")
            + "</th>"
        )
        for col_index in range(compressed_shap_matrix.shape[0]):
            out += (
                '<th style="background:'
                + input_colors[col_index][row_index]
                + '">'
                + str(round(compressed_shap_matrix[col_index][row_index], 3))
                + "</th>"
            )
        out += "</tr>"

    out += "</table>"

    saliency_plot_html = f"""
        <div id="{uuid}_saliency_plot" class="{uuid}_viz_content">
            <div style="margin:5px;font-family:sans-serif;font-weight:bold;">
                <span style="font-size: 20px;"> Saliency Plot </span>
                <br>
                x-axis: Output Text
                <br>
                y-axis: Input Text
            </div>
            {out}
        </div>
    """
    return saliency_plot_html


def heatmap(shap_values):

    # constants

    TREE_NODE_KEY_TOKENS = "tokens"
    TREE_NODE_KEY_CHILDREN = "children"

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    def get_color(shap_value, cmax):
        scaled_value = 0.5 + 0.5 * shap_value / cmax
        color = red_transparent_blue(scaled_value)
        color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
        return color

    def process_text_to_text_shap_values(shap_values):
        processed_values = []

        unpacked_values, clustering = unpack_shap_explanation_contents(
            shap_values
        )
        max_val = 0

        for index, output_token in enumerate(shap_values.output_names):
            (
                tokens,
                values,
                group_sizes,
                token_id_to_node_id_mapping,
                collapsed_node_ids,
            ) = process_shap_values(
                shap_values.data,
                unpacked_values[:, index],
                1,
                "",
                clustering,
                True,
            )
            processed_value = {
                "tokens": tokens,
                "values": values,
                "group_sizes": group_sizes,
                "token_id_to_node_id_mapping": token_id_to_node_id_mapping,
                "collapsed_node_ids": collapsed_node_ids,
            }

            processed_values.append(processed_value)
            max_val = max(max_val, np.max(values))

        return processed_values, max_val

    # unpack input tokens and output tokens
    model_input = shap_values.data
    model_output = shap_values.output_names

    processed_values, max_val = process_text_to_text_shap_values(shap_values)

    # generate dictionary containing precomputed background colors and shap values which are addressable by html token ids
    colors_dict = {}
    shap_values_dict = {}
    token_id_to_node_id_mapping = {}
    cmax = max(
        abs(shap_values.values.min()), abs(shap_values.values.max()), max_val
    )

    # input token -> output token color and label value mapping

    for row_index in range(len(model_input)):
        color_values = {}
        shap_values_list = {}

        for col_index in range(len(model_output)):
            color_values[uuid + "_output_flat_token_" + str(col_index)] = (
                "rgba"
                + str(
                    get_color(shap_values.values[row_index][col_index], cmax)
                )
            )
            shap_values_list[
                uuid + "_output_flat_value_label_" + str(col_index)
            ] = round(shap_values.values[row_index][col_index], 3)

        colors_dict[f"{uuid}_input_node_{row_index}_content"] = color_values
        shap_values_dict[
            f"{uuid}_input_node_{row_index}_content"
        ] = shap_values_list

    # output token -> input token color and label value mapping

    for col_index in range(len(model_output)):
        color_values = {}
        shap_values_list = {}

        for row_index in range(
            processed_values[col_index]["collapsed_node_ids"].shape[0]
        ):
            color_values[
                uuid
                + "_input_node_"
                + str(
                    processed_values[col_index]["collapsed_node_ids"][
                        row_index
                    ]
                )
                + "_content"
            ] = (
                "rgba"
                + str(
                    get_color(
                        processed_values[col_index]["values"][row_index], cmax
                    )
                )
            )
            shap_label_value_str = str(
                round(processed_values[col_index]["values"][row_index], 3)
            )
            if processed_values[col_index]["group_sizes"][row_index] > 1:
                shap_label_value_str += "/" + str(
                    processed_values[col_index]["group_sizes"][row_index]
                )

            shap_values_list[
                uuid
                + "_input_node_"
                + str(
                    processed_values[col_index]["collapsed_node_ids"][
                        row_index
                    ]
                )
                + "_label"
            ] = shap_label_value_str

        colors_dict[
            uuid + "_output_flat_token_" + str(col_index)
        ] = color_values
        shap_values_dict[
            uuid + "_output_flat_token_" + str(col_index)
        ] = shap_values_list

        token_id_to_node_id_mapping_dict = {}

        for index, node_id in enumerate(
            processed_values[col_index]["token_id_to_node_id_mapping"].tolist()
        ):
            token_id_to_node_id_mapping_dict[
                f"{uuid}_input_node_{index}_content"
            ] = f"{uuid}_input_node_{int(node_id)}_content"

        token_id_to_node_id_mapping[
            uuid + "_output_flat_token_" + str(col_index)
        ] = token_id_to_node_id_mapping_dict

    # convert python dictionary into json to be inserted into the runtime javascript environment
    colors_json = json.dumps(colors_dict)
    shap_values_json = json.dumps(shap_values_dict)
    token_id_to_node_id_mapping_json = json.dumps(token_id_to_node_id_mapping)

    javascript_values = (
        "<script> "
        + f"colors_{uuid} = "
        + colors_json
        + "\n"
        + f" shap_values_{uuid} = "
        + shap_values_json
        + "\n"
        + f" token_id_to_node_id_mapping_{uuid} = "
        + token_id_to_node_id_mapping_json
        + "\n"
        + "</script> \n "
    )

    def generate_tree(shap_values):
        num_tokens = len(shap_values.data)
        token_list = {}

        for index in range(num_tokens):
            node_content = {}
            node_content[TREE_NODE_KEY_TOKENS] = shap_values.data[index]
            node_content[TREE_NODE_KEY_CHILDREN] = {}
            token_list[str(index)] = node_content

        counter = num_tokens
        for pair in shap_values.clustering:
            first_node = str(int(pair[0]))
            second_node = str(int(pair[1]))

            new_node_content = {}
            new_node_content[TREE_NODE_KEY_CHILDREN] = {
                first_node: token_list[first_node],
                second_node: token_list[second_node],
            }

            token_list[str(counter)] = new_node_content
            counter += 1

            del token_list[first_node]
            del token_list[second_node]

        return token_list

    tree = generate_tree(shap_values)

    # generates the input token html elements
    # each element contains the label value (initially hidden) and the token text

    input_text_html = ""

    def populate_input_tree(input_index, token_list_subtree, input_text_html):
        content = token_list_subtree[input_index]
        input_text_html += f'<div id="{uuid}_input_node_{input_index}_container" style="display:inline;text-align:center">'

        input_text_html += f'<div id="{uuid}_input_node_{input_index}_label" style="display:none; padding-top: 0px; font-size:12px;">'

        input_text_html += "</div>"

        if token_list_subtree[input_index][TREE_NODE_KEY_CHILDREN]:
            input_text_html += f'<div id="{uuid}_input_node_{input_index}_content" style="display:inline;">'
            for child_index, child_content in token_list_subtree[input_index][
                TREE_NODE_KEY_CHILDREN
            ].items():
                input_text_html = populate_input_tree(
                    child_index,
                    token_list_subtree[input_index][TREE_NODE_KEY_CHILDREN],
                    input_text_html,
                )
            input_text_html += "</div>"
        else:
            input_text_html += (
                f'<div id="{uuid}_input_node_{input_index}_content"'
                + "style='display: inline; background:transparent; border-radius: 3px; padding: 0px;cursor: default;cursor: pointer;'"
                + f'onmouseover="onMouseHoverFlat_{uuid}(this.id)" '
                + f'onmouseout="onMouseOutFlat_{uuid}(this.id)" '
                + f'onclick="onMouseClickFlat_{uuid}(this.id)" '
                + ">"
            )
            input_text_html += (
                content[TREE_NODE_KEY_TOKENS]
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ##", "")
                .replace("▁", "")
                .replace("Ġ", "")
            )
            input_text_html += "</div>"

        input_text_html += "</div>"

        return input_text_html

    input_text_html = populate_input_tree(
        list(tree.keys())[0], tree, input_text_html
    )

    # generates the output token html elements
    output_text_html = ""

    for i in range(len(model_output)):
        output_text_html += (
            "<div style='display:inline; text-align:center;'>"
            + f"<div id='{uuid}_output_flat_value_label_"
            + str(i)
            + "'"
            + "style='display:none;color: #999; padding-top: 0px; font-size:12px;'>"
            + "</div>"
            + f"<div id='{uuid}_output_flat_token_"
            + str(i)
            + "'"
            + "style='display: inline; background:transparent; border-radius: 3px; padding: 0px;cursor: default;cursor: pointer;'"
            + f'onmouseover="onMouseHoverFlat_{uuid}(this.id)" '
            + f'onmouseout="onMouseOutFlat_{uuid}(this.id)" '
            + f'onclick="onMouseClickFlat_{uuid}(this.id)" '
            + ">"
            + model_output[i]
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace(" ##", "")
            .replace("▁", "")
            .replace("Ġ", "")
            + " </div>"
            + "</div>"
        )

    heatmap_html = f"""
        <div id="{uuid}_heatmap" class="{uuid}_viz_content">
          <div id="{uuid}_heatmap_header" style="padding:15px;margin:5px;font-family:sans-serif;font-weight:bold;">
            <div style="display:inline">
              <span style="font-size: 20px;"> Input/Output - Heatmap </span>
            </div>
            <div style="display:inline;float:right">
              Layout :
              <select name="alignment" id="{uuid}_alignment" onchange="selectAlignment_{uuid}(this)">
                <option value="left-right" selected="selected">Left/Right</option>
                <option value="top-bottom">Top/Bottom</option>
              </select>
            </div>
          </div>
          <div id="{uuid}_heatmap_content" style="display:flex;">
            <div id="{uuid}_input_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
              <div id="{uuid}_input_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
                Input Text
              </div>
              <div id="{uuid}_input_content" style="margin:5px;font-family:sans-serif;">
                  {input_text_html}
              </div>
            </div>
            <div id="{uuid}_output_container" style="padding:15px;border-style:solid;margin:5px;flex:1;">
              <div id="{uuid}_output_header" style="margin:5px;font-weight:bold;font-family:sans-serif;margin-bottom:10px">
                Output Text
              </div>
              <div id="{uuid}_output_content" style="margin:5px;font-family:sans-serif;">
                  {output_text_html}
              </div>
            </div>
          </div>
        </div>
    """

    heatmap_javascript = f"""
        <script>
            function selectAlignment_{uuid}(selectObject) {{
                var value = selectObject.value;
                if ( value === "left-right" ){{
                  document.getElementById('{uuid}_heatmap_content').style.display  = "flex";
                }}
                else if ( value === "top-bottom" ) {{
                  document.getElementById('{uuid}_heatmap_content').style.display  = "inline";
                }}
            }}

            var {uuid}_heatmap_flat_state = null;

            function onMouseHoverFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    setBackgroundColors_{uuid}(id);
                    document.getElementById(id).style.backgroundColor  = "grey";
                }}

                if (getIdSide_{uuid}(id) === 'input' && getIdSide_{uuid}({uuid}_heatmap_flat_state) === 'output'){{

                    label_content_id = token_id_to_node_id_mapping_{uuid}[{uuid}_heatmap_flat_state][id];

                    if (document.getElementById(label_content_id).previousElementSibling.style.display == 'none'){{
                        document.getElementById(label_content_id).style.textShadow = "0px 0px 1px #000000";
                    }}

                }}

            }}

            function onMouseOutFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    cleanValuesAndColors_{uuid}(id);
                    document.getElementById(id).style.backgroundColor  = "transparent";
                }}

                if (getIdSide_{uuid}(id) === 'input' && getIdSide_{uuid}({uuid}_heatmap_flat_state) === 'output'){{

                    label_content_id = token_id_to_node_id_mapping_{uuid}[{uuid}_heatmap_flat_state][id];

                    if (document.getElementById(label_content_id).previousElementSibling.style.display == 'none'){{
                        document.getElementById(label_content_id).style.textShadow = "inherit";
                    }}

                }}

            }}

            function onMouseClickFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === id) {{

                    // If the clicked token was already selected

                    document.getElementById(id).style.backgroundColor  = "transparent";
                    cleanValuesAndColors_{uuid}(id);
                    {uuid}_heatmap_flat_state = null;
                }}
                else {{
                    if ({uuid}_heatmap_flat_state === null) {{

                        // No token previously selected, new token clicked on

                        cleanValuesAndColors_{uuid}(id)
                        {uuid}_heatmap_flat_state = id;
                        document.getElementById(id).style.backgroundColor  = "grey";
                        setLabelValues_{uuid}(id);
                        setBackgroundColors_{uuid}(id);
                    }}
                    else {{
                        if (getIdSide_{uuid}({uuid}_heatmap_flat_state) === getIdSide_{uuid}(id)) {{

                            // User clicked a token on the same side as the currently selected token

                            cleanValuesAndColors_{uuid}({uuid}_heatmap_flat_state)
                            document.getElementById({uuid}_heatmap_flat_state).style.backgroundColor  = "transparent";
                            {uuid}_heatmap_flat_state = id;
                            document.getElementById(id).style.backgroundColor  = "grey";
                            setLabelValues_{uuid}(id);
                            setBackgroundColors_{uuid}(id);
                        }}
                        else{{

                            if (getIdSide_{uuid}(id) === 'input') {{
                                label_content_id = token_id_to_node_id_mapping_{uuid}[{uuid}_heatmap_flat_state][id];

                                if (document.getElementById(label_content_id).previousElementSibling.style.display == 'none') {{
                                    document.getElementById(label_content_id).previousElementSibling.style.display = 'block';
                                    document.getElementById(label_content_id).parentNode.style.display = 'inline-block';
                                    document.getElementById(label_content_id).style.textShadow = "0px 0px 1px #000000";
                                  }}
                                else {{
                                    document.getElementById(label_content_id).previousElementSibling.style.display = 'none';
                                    document.getElementById(label_content_id).parentNode.style.display = 'inline';
                                    document.getElementById(label_content_id).style.textShadow  = "inherit";
                                  }}

                            }}
                            else {{
                                if (document.getElementById(id).previousElementSibling.style.display == 'none') {{
                                    document.getElementById(id).previousElementSibling.style.display = 'block';
                                    document.getElementById(id).parentNode.style.display = 'inline-block';
                                  }}
                                else {{
                                    document.getElementById(id).previousElementSibling.style.display = 'none';
                                    document.getElementById(id).parentNode.style.display = 'inline';
                                  }}
                            }}

                        }}
                    }}

                }}
            }}

            function setLabelValues_{uuid}(id) {{
                for(const token in shap_values_{uuid}[id]){{
                    document.getElementById(token).innerHTML = shap_values_{uuid}[id][token];
                    document.getElementById(token).nextElementSibling.title = 'SHAP Value : ' + shap_values_{uuid}[id][token];
                }}
            }}

            function setBackgroundColors_{uuid}(id) {{
                for(const token in colors_{uuid}[id]){{
                    document.getElementById(token).style.backgroundColor  = colors_{uuid}[id][token];
                }}
            }}

            function cleanValuesAndColors_{uuid}(id) {{
                for(const token in shap_values_{uuid}[id]){{
                    document.getElementById(token).innerHTML = "";
                    document.getElementById(token).nextElementSibling.title = "";
                }}
                 for(const token in colors_{uuid}[id]){{
                    document.getElementById(token).style.backgroundColor  = "transparent";
                    document.getElementById(token).previousElementSibling.style.display = 'none';
                    document.getElementById(token).parentNode.style.display = 'inline';
                    document.getElementById(token).style.textShadow  = "inherit";
                }}
            }}

            function getIdSide_{uuid}(id) {{
                if (id === null) {{
                    return 'null'
                }}
                return id.split("_")[1];
            }}
        </script>
    """

    return heatmap_html + heatmap_javascript + javascript_values


def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return values, clustering
