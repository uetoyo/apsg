import pickle
import warnings
from copy import deepcopy

import numpy as np
from abc import ABC

from apsg._settings import settings

# @fixme circular import

__all__ = ("Vec3",)


from apsg._helpers import (
    KentDistribution,
    sind,
    cosd,
    acosd,
    asind,
    atand,
    atan2d,
    angle_metric,
    l2v,
    getldd,
    _linear_inverse_kamb,
    _square_inverse_kamb,
    _schmidt_count,
    _kamb_count,
    _exponential_kamb,
)


class Vector(ABC):
    """
    The marker base for the verctor types.
    In the future this class will replace the numpy array subclasing.
    """


class Vec3(np.ndarray, Vector):
    """
    ``Vec3`` is base class to store 3-dimensional vectors derived from
    ``numpy.ndarray`` on which ``Lin`` and ``Fol`` classes are based.

    ``Vec3`` support most of common vector algebra using following operators
        - ``+`` - vector addition
        - ``-`` - vector subtraction
        - ``*`` - dot product
        - ``**`` - cross product
        - ``abs`` - magnitude (length) of vector

    Check following methods and properties for additional operations.

    Args:
        arr (array_like):
            Input data that or can be converted to an array.
            This includes lists, tuples, and ndarrays. When more than one
            argument is passed (i.e. `inc` is not `None`) `arr` is interpreted
            as dip direction of the vector in degrees.
        inc (float):
            `None` or dip of the vector in degrees.
        mag (float):
            The magnitude of the vector if `inc` is not `None`.

    Returns:
      ``Vec3`` object

    Example:
      >>> v = Vec3([1, -2, 3])
      >>> abs(v)
      3.7416573867739413

      # The dip direction and dip angle of vector with magnitude of 1 and 3.
      >>> v = Vec3(120, 60)
      >>> abs(v)
      1.0

      >>> v = Vec3(120, 60, 3)
      >>> abs(v)
      3.0
    """

    def __new__(cls, arr, mag=1.0):
        return np.asarray(arr).view(cls)

    def __repr__(self):
        if settings["vec2dd"]:
            result = "V:{:.0f}/{:.0f}".format(*self.dd)
        else:
            result = "V({:.3f}, {:.3f}, {:.3f})".format(*self)
        return result

    def __str__(self):
        return repr(self)

    def __mul__(self, other):
        """
        Return the dot product of two vectors.
        """
        return np.dot(self, other)  # What about `numpy.inner`?

    def __abs__(self):
        """
        Return the 2-norm or Euclidean norm of vector.
        """

        return np.linalg.norm(self)

    def __pow__(self, other):
        """
        Return cross product if argument is vector or power of vector.
        """
        if np.isscalar(other):
            return pow(abs(self), other)
        else:
            return self.cross(other)

    def __eq__(self, other):
        """
        Return `True` if vectors are equal, otherwise `False`.
        """
        if not isinstance(other, self.__class__):
            return False
        return self is other or abs(self - other) < settings["precision"]

    def __ne__(self, other):
        """
        Return `True` if vectors are not equal, otherwise `False`.

        Overrides the default implementation (unnecessary in Python 3).
        """
        return not self == other

    def __hash__(self):
        return NotImplementedError

    @classmethod
    def rand(cls):
        """
        Random unit vector from distribution on sphere
        """
        return cls(np.random.randn(3)).uv

    @property
    def type(self):
        """
        Return the type of ``self``.
        """
        return type(self)

    @property
    def upper(self):
        """
        Return `True` if z-coordinate is negative, otherwise `False`.
        """
        return np.sign(self[2]) < 0

    @property
    def flip(self):
        """
        Return a new vector with inverted `z` coordinate.
        """
        return Vec3((self[0], self[1], -self[2]))

    @property
    def uv(self):
        """
        Normalize the vector to unit length.

        Returns:
          unit vector of ``self``

        Example:
          >>> u = Vec3([1,1,1])
          >>> u.uv
          V(0.577, 0.577, 0.577)

        """

        return self / abs(self)

    def cross(self, other):
        """
        Calculate the cross product of two vectors.

        Args:
            other: other ``Vec3`` vector

        Returns:
             The cross product of `self` and `other`.

        Example:
            >>> v = Vec3([1, 0, 0])
            >>> u = Vec3([0, 0, 1])
            >>> v.cross(u)
            V(0.000, -1.000, 0.000)

        """

        return Vec3(np.cross(self, other))

    def angle(self, other):
        """
        Calculate the angle between two vectors in degrees.

        Args:
            other: other ``Vec3`` vector

        Returns:
            The angle between `self` and `other` in degrees.

        Example:
            >>> v = Vec3([1, 0, 0])
            >>> u = Vec3([0, 0, 1])
            >>> v.angle(u)
            90.0
        """

        # if isinstance(other, Group):
        #     return other.angle(self)
        # else:
        return acosd(np.clip(np.dot(self.uv, other.uv), -1, 1))

    def rotate(self, axis: "Vec3", angle: float):
        """
        Return rotated vector about axis.

        Args:
            axis (``Vec3``): axis of rotation
            angle (float): angle of rotation in degrees

        Returns:
            vector represenatation of `self` rotated `angle` degrees about
            vector `axis`. Rotation is clockwise along axis direction.

        Example:
            # Rotate `e1` vector around `z` axis.
            >>> u = Vec3([1, 0, 0])
            >>> z = Vec3([0, 0, 1])
            >>> u.rotate(z, 90)
            V(0.000, 1.000, 0.000)

        """

        e = Vec3(self)  # rotate all types as vectors
        k = axis.uv
        r = cosd(angle) * e + sind(angle) * k.cross(e) + (1 - cosd(angle)) * k * (k * e)

        return r.view(type(self))

    def proj(self, other):
        """
        Return projection of vector `u` onto vector `v`.

        Args:
            other (``Vec3``): other vector

        Returns:
            vector representation of `self` projected onto 'other'

        Example:
            >> u.proj(v)

        Note:
            To project on plane use: `u - u.proj(v)`, where `v` is plane normal.

        """

        r = np.dot(self, other) * other / np.linalg.norm(other)

        return r.view(type(self))

    # def H(self, other):
    #     """
    #     Return ``DefGrad`` rotational matrix H which rotate vector
    #     `u` to vector `v`. Axis of rotation is perpendicular to both
    #     vectors `u` and `v`.

    #     Args:
    #         other (``Vec3``): other vector

    #     Returns:
    #         ``Defgrad`` rotational matrix

    #     Example:
    #         >>> u = Vec3(210, 50)
    #         >>> v = Vec3(60, 70)
    #         >>> u.transform(u.H(v)) == v
    #         True

    #     """
    #     from apsg.tensors import DefGrad

    #     return DefGrad.from_axis(self ** other, self.V.angle(other))

    def transform(self, F, **kwargs):
        """
        Return affine transformation of vector `u` by matrix `F`.

        Args:
            F (``DefGrad`` or ``numpy.array``): transformation matrix

        Keyword Args:
            norm: normalize transformed vectors. [True or False] Default False

        Returns:
            vector representation of affine transformation (dot product)
            of `self` by `F`

        Example:
            # Reflexion of `y` axis.
            >>> F = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
            >>> u = Vec3([1, 1, 1])
            >>> u.transform(F)
            V(1.000, -1.000, 1.000)

        """
        if kwargs.get("norm", False):
            res = np.dot(F, self).view(type(self)).uv
        else:
            res = np.dot(F, self).view(type(self))
        return res

    @property
    def dd(self):
        """
        Return azimuth, inclination tuple.

        Example:
          >>> v = Vec3([1, 0, -1])
          >>> azi, inc = v.dd
          >>> azi
          0.0
          >>> inc
          -44.99999999999999

        """

        n = self.uv
        azi = atan2d(n[1], n[0]) % 360
        inc = asind(n[2])

        return azi, inc

    # @property
    # def aslin(self):
    #     """
    #     Convert `self` to ``Lin`` object.

    #     Example:
    #         >>> u = Vec3([1,1,1])
    #         >>> u.aslin
    #         L:45/35
    #     """
    #     return self.copy().view(Lin)

    # @property
    # def asfol(self):
    #     """
    #     Convert `self` to ``Fol`` object.

    #     Example:
    #         >>> u = Vec3([1,1,1])
    #         >>> u.asfol
    #         S:225/55
    #     """
    #     return self.copy().view(Fol)

    # @property
    # def asvec3(self):
    #     """
    #     Convert `self` to ``Vec3`` object.

    #     Example:
    #         >>> l = Lin(120,50)
    #         >>> l.asvec3
    #         V(-0.321, 0.557, 0.766)
    #     """
    #     return self.copy().view(Vec3)

    # @property
    # def V(self):
    #     """
    #     Convert `self` to ``Vec3`` object.

    #     Note:
    #         This is an alias of ``asvec3`` property.
    #     """
    #     return self.copy().view(Vec3)
