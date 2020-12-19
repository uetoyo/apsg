# -*- coding: utf-8 -*-

"""
Test the vector module classes and functions.
"""


import pytest
import numpy as np


from apsg import Vec3


class TestVector:
    @pytest.fixture
    def x(self):
        return Vec3([1, 0, 0])

    @pytest.fixture
    def y(self):
        return Vec3([0, 1, 0])

    @pytest.fixture
    def z(self):
        return Vec3([0, 0, 1])

    @pytest.mark.skip
    def test_that_vector_is_hashable(self, helpers):
        assert helpers.is_hashable(Vec3([1, 2, 3]))

    def test_that_vec3_string_gets_three_digits_when_vec2dd_settings_is_false(self):
        settings["vec2dd"] = False

        vec = Vec3([1, 2, 3])

        current = str(vec)
        expects = "V(1.000, 2.000, 3.000)"

        assert current == expects

    def test_that_vec3_string_gets_dip_and_dir_when_vec2dd_settings_is_true(self):
        settings["vec2dd"] = True

        vec = Vec3([1, 2, 3])

        current = str(vec)
        expects = "V:63/53"

        assert current == expects

        settings["vec2dd"] = False

    # ``==`` operator

    def test_that_equality_operator_is_reflexive(self):
        u = Vec3([1, 2, 3])

        assert u == u

    def test_that_equality_operator_is_symetric(self):
        u = Vec3([1, 2, 3])
        v = Vec3([1, 2, 3])

        assert u == v and v == u

    def test_that_equality_operator_is_transitive(self):
        u = Vec3([1, 2, 3])
        v = Vec3([1, 2, 3])
        w = Vec3([1, 2, 3])

        assert u == v and v == w and u == w

    def test_that_equality_operator_precision_limits(self):
        """
        This is not the best method how to test a floating point precision limits,
        but I will keep it here for a future work.
        """
        lhs = Vec3([1.00000000000000001] * 3)
        rhs = Vec3([1.00000000000000009] * 3)

        assert lhs == rhs

    def test_that_equality_operator_returns_false_for_none(self):
        lhs = Vec3([1, 0, 0])
        rhs = None

        current = lhs == rhs
        expects = False

        assert current == expects

    # ``!=`` operator

    def test_inequality_operator(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([3, 2, 1])

        assert lhs != rhs

    # ``hash`` method

    @pytest.mark.skip
    def test_that_hash_is_same_for_identical_vectors(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([1, 2, 3])

        assert hash(lhs) == hash(rhs)

    @pytest.mark.skip
    def test_that_hash_is_not_same_for_different_vectors(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([3, 2, 1])

        assert not hash(lhs) == hash(rhs)

    # ``upper`` property

    def test_that_vector_is_upper(self):
        vec = Vec3([0, 0, -1])

        assert vec.upper

    def test_that_vector_is_not_upper(self):
        vec = Vec3([0, 0, 1])

        assert not vec.upper

    # ``flip`` property

    def test_that_vector_is_flipped(self):
        current = Vec3([0, 0, 1]).flip
        expects = Vec3([0, 0, -1])

        assert current == expects

    # ``abs`` operator

    def test_absolute_value(self):
        current = abs(Vec3([1, 2, 3]))
        expects = 3.7416573867739413

        assert current == expects

    # ``uv`` property

    def test_that_vector_is_normalized(self):
        current = Vec3([1, 2, 3]).uv
        expects = Vec3([0.26726124191242442, 0.5345224838248488, 0.8017837257372732])

        assert current == expects

    # ``dd`` property

    def test_dd_property(self):
        v = Vec3([1, 0, 0])

        current = v.dd
        expects = (0.0, 0.0)

        assert current == expects

    # ``aslin`` property

    def test_aslin_conversion(self):
        assert str(Vec3([1, 1, 1]).aslin) == str(Lin(45, 35))  # `Vec` to `Lin`
        assert str(Vec3(Lin(110, 37)).aslin) == str(
            Lin(110, 37)
        )  # `Lin` to `Vec` to `Lin`

    # ``asfol`` property

    def test_asfol_conversion(self):
        assert str(Vec3([1, 1, 1]).asfol) == str(Fol(225, 55))  # `Vec` to `Fol`
        assert str(Vec3(Fol(213, 52)).asfol) == str(
            Fol(213, 52)
        )  # `Fol` to `Vec` to `Fol`

    # ``asvec`` property

    def test_asvec_conversion(self):
        assert str(Lin(120, 10).asvec3) == str(Vec3(120, 10, 1))

    # ``angle`` property

    def test_that_angle_between_vectors_is_0_degrees_when_they_are_collinear(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([2, 0, 0])

        current = lhs.angle(rhs)
        expects = 0

        assert current == expects

    def test_that_angle_between_vectors_is_90_degrees_when_they_are_perpendicular(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 1])

        current = lhs.angle(rhs)
        expects = 90  # degrees

        assert current == expects

    def test_that_angle_between_vectors_is_180_degrees_when_they_are_opposite(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([-1, 0, 0])

        current = lhs.angle(rhs)
        expects = 180  # degrees

        assert current == expects

    # ``cross`` method

    def test_that_vector_product_is_anticommutative(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 0])

        assert lhs.cross(rhs) == -rhs.cross(lhs)

    def test_that_vector_product_is_distributive_over_addition(self):
        a = Vec3([1, 0, 0])
        b = Vec3([0, 1, 0])
        c = Vec3([0, 0, 1])

        assert a.cross(b + c) == a.cross(b) + a.cross(c)

    def test_that_vector_product_is_zero_vector_when_they_are_collinear(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([2, 0, 0])

        current = lhs.cross(rhs)
        expects = Vec3([0, 0, 0])

        assert current == expects

    def test_that_vector_product_is_zero_vector_when_they_are_opposite(self):

        lhs = Vec3([1, 0, 0])
        rhs = Vec3([-1, 0, 0])

        current = lhs.cross(rhs)
        expects = Vec3([0, 0, 0])

        assert current == expects

    def test_vector_product_of_orthonormal_vectors(self):
        e1 = Vec3([1, 0, 0])
        e2 = Vec3([0, 1, 0])

        current = e1.cross(e2)
        expects = Vec3([0, 0, 1])

        assert current == expects

    # ``dot`` method

    def test_scalar_product_of_same_vectors(self):
        i = Vec3([1, 2, 3])

        assert np.allclose(i.dot(i), abs(i) ** 2)

    def test_scalar_product_of_orthonornal_vectors(self):
        i = Vec3([1, 0, 0])
        j = Vec3([0, 1, 0])

        assert i.dot(j) == 0

    # ``rotate`` method

    def test_rotation_by_90_degrees_around_axis(self, z):
        v = Vec3([1, 1, 1])
        current = v.rotate(z, 90)
        expects = Vec3([-1, 1, 1])

        assert current == expects

    def test_rotation_by_180_degrees_around_axis(self, z):
        v = Vec3([1, 1, 1])
        current = v.rotate(z, 180)
        expects = Vec3([-1, -1, 1])

        assert current == expects

    def test_rotation_by_360_degrees_around_axis(self, z):
        v = Vec3([1, 1, 1])
        current = v.rotate(z, 360)
        expects = Vec3([1, 1, 1])

        assert current == expects

    # ``proj`` method

    def test_projection_of_xy_onto(self, z):
        xz = Vec3([1, 0, 1])
        current = xz.proj(z)
        expects = Vec3([0, 0, 1])

        assert current == expects

    # ``H`` method

    def test_mutual_rotation(self, x, y, z):
        current = x.H(y)
        expects = DefGrad.from_axis(z, 90)

        assert current == expects

    # ``transform`` method

    def test_transform_method(self, x, y, z):
        F = DefGrad.from_axis(z, 90)
        current = x.transform(F)
        expects = y

        assert current == expects

    def test_add_operator(self):
        lhs = Vec3([1, 1, 1])
        rhs = Vec3([1, 1, 1])

        current = lhs + rhs
        expects = Vec3([2, 2, 2])

        assert current == expects

    def test_sub_operator(self):
        lhs = Vec3([1, 2, 3])
        rhs = Vec3([3, 1, 2])

        current = lhs - rhs
        expects = Vec3([-2, 1, 1])

        assert current == expects

    # ``*`` operator aka dot product

    def test_mull_operator(self):
        lhs = Vec3([1, 1, 1])
        rhs = Vec3([1, 1, 1])

        current = lhs * rhs
        expects = lhs.dot(rhs)

        assert np.allclose(current, expects)

    # ``**`` operator aka cross product

    def test_pow_operator_with_vector(self):
        lhs = Vec3([1, 0, 0])
        rhs = Vec3([0, 1, 0])

        current = lhs ** rhs
        expects = lhs.cross(rhs)

        assert current == expects

    def test_pow_operator_with_scalar(self):
        lhs = Vec3([1, 1, 1])
        rhs = 2

        current = lhs ** rhs
        expects = np.dot(lhs, lhs)

        assert np.allclose(current, expects)

    def test_length_method(self):
        w = Vec3([1, 2, 3])

        assert len(w) == 3

    def test_getitem_operator(self):
        v = Vec3([1, 2, 3])

        assert all((v[0] == 1, v[1] == 2, v[2] == 3))
