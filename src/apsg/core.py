# -*- coding: utf-8 -*-

"""
A module to manipulate, analyze and visualize structural geology data.
"""

import pickle
import warnings
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

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

from abc import ABC
from apsg._feature import Group, Lin


__all__ = (
    "Cluster",
    "StereoGrid",
)


class Cluster:
    """
    Provides a hierarchical clustering using `scipy.cluster` routines.

    The distance matrix is calculated as an angle between features, where ``Fol`` and
    ``Lin`` use axial angles while ``Vec3`` uses direction angles.
    """

    def __init__(self, d, **kwargs):
        assert isinstance(d, Group), "Only group could be clustered"
        self.data = Group(d.copy())
        self.maxclust = kwargs.get("maxclust", 2)
        self.angle = kwargs.get("angle", None)
        self.method = kwargs.get("method", "average")
        self.pdist = self.data.angle()
        self.linkage()

    def __repr__(self):
        if hasattr(self, "groups"):
            info = "Already %d clusters created." % len(self.groups)
        else:
            info = "Not yet clustered. Use cluster() method."
        if self.angle is not None:
            crit = "Criterion: Angle\nSettings: angle=%.4g\n" % (self.angle)
        else:
            crit = "Criterion: Maxclust\nSettings: muxclust=%.4g\n" % (self.maxclust)
        return (
            "Clustering object\n"
            + "Number of data: %d\n" % len(self.data)
            + "Linkage method: %s\n" % self.method
            + crit
            + info
        )

    def cluster(self, **kwargs):
        """Do clustering on data

        Result is stored as tuple of Groups in ``groups`` property.

        Keyword Args:
          criterion: The criterion to use in forming flat clusters
          maxclust: number of clusters
          angle: maximum cophenetic distance(angle) in clusters
        """
        from scipy.cluster.hierarchy import fcluster

        self.maxclust = kwargs.get("maxclust", 2)
        self.angle = kwargs.get("angle", None)
        if self.angle is not None:
            self.idx = fcluster(self.Z, self.angle, criterion="distance")
        else:
            self.idx = fcluster(self.Z, self.maxclust, criterion="maxclust")
        self.groups = tuple(
            self.data[np.flatnonzero(self.idx == c)] for c in np.unique(self.idx)
        )

    def linkage(self, **kwargs):
        """Do linkage of distance matrix

        Keyword Args:
          method: The linkage algorithm to use
        """
        from scipy.cluster.hierarchy import linkage

        self.method = kwargs.get("method", "average")
        self.Z = linkage(self.pdist, method=self.method, metric=angle_metric)

    def dendrogram(self, **kwargs):
        """Show dendrogram

        See ``scipy.cluster.hierarchy.dendrogram`` for possible kwargs.
        """
        from scipy.cluster.hierarchy import dendrogram

        fig, ax = plt.subplots(figsize=settings["figsize"])
        dendrogram(self.Z, ax=ax, **kwargs)
        plt.show()

    def elbow(self, no_plot=False, n=None):
        """Plot within groups variance vs. number of clusters.

        Elbow criterion could be used to determine number of clusters.
        """
        from scipy.cluster.hierarchy import fcluster

        if n is None:
            idx = fcluster(self.Z, len(self.data), criterion="maxclust")
            nclust = list(np.arange(1, np.sqrt(idx.max() / 2) + 1, dtype=int))
        else:
            nclust = list(np.arange(1, n + 1, dtype=int))
        within_grp_var = []
        mean_var = []
        for n in nclust:
            idx = fcluster(self.Z, n, criterion="maxclust")
            grp = [np.flatnonzero(idx == c) for c in np.unique(idx)]
            # between_grp_var = Group([self.data[ix].R.uv for ix in grp]).var
            var = [100 * self.data[ix].var for ix in grp]
            within_grp_var.append(var)
            mean_var.append(np.mean(var))
        if not no_plot:
            fig, ax = plt.subplots(figsize=settings["figsize"])
            ax.boxplot(within_grp_var, positions=nclust)
            ax.plot(nclust, mean_var, "k")
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Variance")
            ax.set_title("Within-groups variance vs. number of clusters")
            plt.show()
        else:
            return nclust, within_grp_var

    @property
    def R(self):
        """Return group of clusters resultants."""
        return Group([group.R for group in self.groups])


class StereoGrid:
    """
    The class to store regular grid of values to be contoured on ``StereoNet``.

    ``StereoGrid`` object could be calculated from ``Group`` object or by user-
    defined function, which accept unit vector as argument.

    Args:
      g: ``Group`` object of data to be used for desity calculation. If
      ommited, zero values grid is returned.

    Keyword Args:
      npoints: approximate number of grid points Default 1800
      grid: type of grid 'radial' or 'ortho'. Default 'radial'
      sigma: sigma for kernels. Default 1
      method: 'exp_kamb', 'linear_kamb', 'square_kamb', 'schmidt', 'kamb'.
        Default 'exp_kamb'
      trim: Set negative values to zero. Default False
      Note: Euclidean norms are used as weights. Normalize data if you dont want to use weigths.

    """

    def __init__(self, d=None, **kwargs):
        self.initgrid(**kwargs)
        if d:
            assert isinstance(d, Group), "StereoGrid need Group as argument"
            self.calculate_density(np.asarray(d), **kwargs)

    def __repr__(self):
        return (
            "StereoGrid with %d points.\n" % self.n
            + "Maximum: %.4g at %s\n" % (self.max, self.max_at)
            + "Minimum: %.4g at %s" % (self.min, self.min_at)
        )

    @property
    def min(self):
        return self.values.min()

    @property
    def max(self):
        return self.values.max()

    @property
    def min_at(self):
        return Vec3(self.dcgrid[self.values.argmin()]).aslin

    @property
    def max_at(self):
        return Vec3(self.dcgrid[self.values.argmax()]).aslin

    def initgrid(self, **kwargs):
        import matplotlib.tri as tri

        # parse options
        grid = kwargs.get("grid", "radial")
        if grid == "radial":
            ctn_points = int(
                np.round(np.sqrt(kwargs.get("npoints", 1800)) / 0.280269786)
            )
            # calc grid
            self.xg = 0
            self.yg = 0
            for rho in np.linspace(0, 1, int(np.round(ctn_points / 2 / np.pi))):
                theta = np.linspace(0, 360, int(np.round(ctn_points * rho + 1)))[:-1]
                self.xg = np.hstack((self.xg, rho * sind(theta)))
                self.yg = np.hstack((self.yg, rho * cosd(theta)))
        elif grid == "ortho":
            n = int(np.round(np.sqrt(kwargs.get("npoints", 1800) - 4) / 0.8685725142))
            x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
            d2 = (x ** 2 + y ** 2) <= 1
            self.xg = np.hstack((0, 1, 0, -1, x[d2]))
            self.yg = np.hstack((1, 0, -1, 0, y[d2]))
        else:
            raise TypeError("Wrong grid type!")
        self.dcgrid = l2v(*getldd(self.xg, self.yg)).T
        self.n = self.dcgrid.shape[0]
        self.values = np.zeros(self.n, dtype=np.float)
        self.triang = tri.Triangulation(self.xg, self.yg)

    def calculate_density(self, dcdata, **kwargs):
        """Calculate density of elements from ``Group`` object."""
        # parse options
        sigma = kwargs.get("sigma", 1 / len(dcdata) ** (-1 / 7))
        method = kwargs.get("method", "exp_kamb")
        trim = kwargs.get("trim", False)

        func = {
            "linear_kamb": _linear_inverse_kamb,
            "square_kamb": _square_inverse_kamb,
            "schmidt": _schmidt_count,
            "kamb": _kamb_count,
            "exp_kamb": _exponential_kamb,
        }[method]

        # weights are given by euclidean norms of data
        weights = np.linalg.norm(dcdata, axis=1)
        weights /= weights.mean()
        for i in range(self.n):
            dist = np.abs(np.dot(self.dcgrid[i], dcdata.T))
            count, scale = func(dist, sigma)
            count *= weights
            self.values[i] = (count.sum() - 0.5) / scale
        if trim:
            self.values[self.values < 0] = 0

    def apply_func(self, func, *args, **kwargs):
        """Calculate values using function passed as argument.
        Function must accept Vec3-like (or 3 elements array)
        as argument and return scalar value.

        """
        for i in range(self.n):
            self.values[i] = func(Vec3(self.dcgrid[i]), *args, **kwargs)

    def contourf(self, *args, **kwargs):
        """ Show filled contours of values."""
        fig, ax = plt.subplots(figsize=settings["figsize"])
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.tricontourf(self.triang, self.values, *args, **kwargs)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    def contour(self, *args, **kwargs):
        """ Show contours of values."""
        fig, ax = plt.subplots(figsize=settings["figsize"])
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.tricontour(self.triang, self.values, *args, **kwargs)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    def plotcountgrid(self):
        """ Show counting grid."""
        fig, ax = plt.subplots(figsize=settings["figsize"])
        # Projection circle
        ax.text(0, 1.02, "N", ha="center", va="baseline", fontsize=16)
        ax.add_artist(plt.Circle((0, 0), 1, color="w", zorder=0))
        ax.add_artist(plt.Circle((0, 0), 1, color="None", ec="k", zorder=3))
        ax.set_aspect("equal")
        plt.triplot(self.triang, "bo-")
        plt.axis("off")
        plt.show()
