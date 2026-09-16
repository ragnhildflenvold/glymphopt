import abc
from typing import Optional, Self

from click import Option

import dolfin as df
from dolfin import inner, grad, dot


def mass_matrix(V: df.FunctionSpace, dx: Optional[df.Measure] = None):
    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = dx or df.Measure("dx", V.mesh())
    return df.assemble(inner(u, v) * dx)


def stiffness_matrix(V: df.FunctionSpace):
    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = df.Measure("dx", V.mesh())
    return df.assemble(inner(grad(u), grad(v)) * dx)


def boundary_mass_matrix(V: df.FunctionSpace):
    u, v = df.TrialFunction(V), df.TestFunction(V)
    ds = df.Measure("ds", V.mesh())
    return df.assemble(inner(u, v) * ds)


def matmul(M: df.cpp.la.Matrix, x: df.cpp.la.Matrix):
    Mx = df.Vector()
    M.mult(x, Mx)
    return Mx


def domain_integral(V: df.FunctionSpace, dx: Optional[df.Measure] = None):
    dx = dx or df.Measure("dx", domain=V.mesh())
    v = df.TestFunction(V)
    m = df.assemble(v * dx)

    def call(x: df.cpp.la.Vector):
        return m.inner(x)

    return call


def matrix_operator(M: df.cpp.la.Matrix):
    x_ = df.Vector()

    def call(x: df.cpp.la.Vector):
        M.mult(x, x_)
        return x_

    return call


def bilinear_operator(M: df.cpp.la.Matrix):
    Ax = df.Vector()

    def call(x: df.cpp.la.Vector, y: df.cpp.la.Vector):
        M.mult(x, Ax)
        return Ax.inner(y)

    return call


def mass_matrices(W, dx=None):
    dx = dx or df.Measure("dx", W.mesh())
    ce, cp = df.TrialFunctions(W)
    ve, vp = df.TestFunctions(W)
    Me = df.assemble(ce * ve * dx)
    Mp = df.assemble(cp * vp * dx)
    return Me, Mp


def boundary_mass_matrices(W, ds=None):
    ds = ds or df.Measure("ds", W.mesh())
    ce, cp = df.TrialFunctions(W)
    ve, vp = df.TestFunctions(W)
    Be = df.assemble(ce * ve * ds)
    Bp = df.assemble(cp * vp * ds)
    return Be, Bp


def diffusion_operator_matrices(D, W, dx=None):
    dx = dx or df.Measure("dx", W.mesh())
    ce, cp = df.TrialFunctions(W)
    ve, vp = df.TestFunctions(W)
    DKe = df.assemble(inner(dot(D, grad(ce)), grad(ve)) * dx)
    DKp = df.assemble(inner(dot(D, grad(cp)), grad(vp)) * dx)
    return DKe, DKp


def transfer_matrix(W, dx=None):
    dx = dx or df.Measure("dx", W.mesh())
    ce, cp = df.TrialFunctions(W)
    ve, vp = df.TestFunctions(W)
    T = df.assemble(inner(ce - cp, ve - vp) * dx)
    return T


def reaction_matrix(W, dx=None):
    dx = dx or df.Measure("dx", W.mesh())
    _, cp = df.TrialFunctions(W)
    _, vp = df.TestFunctions(W)
    R = df.assemble(inner(cp, vp) * dx)
    return R


class UpdatableCoefficient(abc.ABC):
    def update(self, t: float, *args) -> Self:
        return self

    def __call__(self, t: float, *args) -> df.Vector:
        pass


class UpdatableFunction(df.Function, UpdatableCoefficient):
    pass


class UpdatableExpression(df.Expression, UpdatableCoefficient):
    pass


def zero_vector(V: df.FunctionSpace) -> df.Vector:
    v = df.TestFunction(V)
    ds = df.Measure("dx", V)
    return df.assemble(inner(df.Constant(0.0), v) * ds)
