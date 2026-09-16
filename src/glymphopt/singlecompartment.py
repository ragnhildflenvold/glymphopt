import numpy as np
import dolfin as df
import tqdm
from dolfin import inner, grad

from glymphopt.cache import CacheObject, cache_fetch
from glymphopt.datageneration import BoundaryConcentration
from glymphopt.measure import LossFunction, measure, measure_interval
from glymphopt.operators import (
    matrix_operator,
    bilinear_operator,
    matmul,
)
from glymphopt.timestepper import TimeStepper


class SingleCompartmentModel:
    def __init__(self, V, D, g):
        domain = V.mesh()
        dx = df.Measure("dx", domain)
        ds = df.Measure("ds", domain)

        u, v = df.TrialFunction(V), df.TestFunction(V)
        self.M_ = df.assemble(inner(u, v) * dx)
        self.DK = df.assemble(inner(D * grad(u), grad(v)) * dx)
        self.S = df.assemble(inner(u, v) * ds)
        self.g = g or BoundaryConcentration(V)

    def M(self, coefficients) -> df.Matrix:
        return self.M_

    def L(self, coefficients) -> df.Matrix:
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        return a * self.DK + r * self.M_ + k * self.S


class SingleCompartmentInverseProblem:
    def __init__(
        self,
        td,
        Yd,
        coefficientvector,
        g,
        D,
        dt=3600,
        solver_type="direct",
        timescale=1.0,
        progress=True,
    ):
        self.silent = not progress
        self.td, self.Yd = td, Yd

        t_start = self.td[0]
        dt = dt * timescale
        N = int(np.ceil(np.round((self.td[-1] - t_start) / dt, 12)))
        t_end = N * dt
        self.timestepper = TimeStepper(dt, (t_start, t_end))

        self.V = self.Yd[0].function_space()
        self.model = SingleCompartmentModel(self.V, g=g, D=D)
        self.cache = {
            "state": CacheObject(),
            "adjoint": CacheObject(),
            "g": CacheObject(),
            "sensitivity": CacheObject(),
            "operator": CacheObject(),
        }
        self.solver_type = solver_type

        self.coefficients = coefficientvector
        _M_ = bilinear_operator(self.model.M("_"))
        self.Yd_norms = [_M_(yi.vector(), yi.vector()) for yi in self.Yd]
        self.loss = LossFunction(td, Yd)

    def measure(self, Y: list[df.Function]) -> list[df.Function]:
        return measure(self.timestepper, Y, self.td)

    def F(self, x):
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)
        return self.loss(Ym)

    def forward(self, x):
        coefficients = self.coefficients.from_vector(x)

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()
        Y = [df.Function(self.V, name="state") for _ in range(len(timepoints))]
        Y[0].assign(self.Yd[0])

        model = self.model
        M = model.M(coefficients)
        L = model.L(coefficients)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"x": x}, x=x)

        if self.solver_type == "direct":
            solver = cache_fetch(
                self.cache["operator"],
                df.LUSolver,
                {"x": x, "solver": self.solver_type},
                A=M + (0.5 * dt) * L,  # Crank-Nicolson
                # A=M + dt * L, # Implicit Euler
            )
        else:
            solver = cache_fetch(
                self.cache["operator"],
                df.KrylovSolver,
                {"x": x, "solver": self.solver_type},
                A=M + (0.5 * dt) * L,  # Crank-Nicolson
                method="cg",
                preconditioner="jacobi",
            )
        Mdot = matrix_operator(M - (0.5 * dt) * L)  # Crank-Nicolson
        # Mdot = matrix_operator(M)  # Implicit Euler

        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(
                Y[n + 1].vector(),
                Mdot(Y[n].vector()) + 0.5 * dt * (G[n] + G[n + 1]),  # Crank-Nicolson
                # Mdot(Y[n].vector()) + (dt) * G[n + 1],  # Implicit-Euler
            )
        return Y

    def boundary_vectors(self, x):
        coefficients = self.coefficients.from_vector(x)
        eta = coefficients["eta"]
        k = coefficients["k"]

        model = self.model
        timestepper = self.timestepper
        return [k * eta * matmul(model.S, model.g(t)) for t in timestepper.vector()]

    def gradF(self, x):
        coefficients = self.coefficients.from_vector(x)
        k = coefficients["k"]
        eta = coefficients["eta"]
        dt = self.timestepper.dt
        model = self.model
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)

        P = cache_fetch(self.cache["adjoint"], self.adjoint, {"x": x}, x=x, Ym=Ym)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"eta": eta}, eta=eta)

        _M_ = bilinear_operator(model.M)
        _DK_ = bilinear_operator(model.DK)
        _S_ = bilinear_operator(model.S)
        return dt * sum(
            np.array(
                [
                    _DK_(p.vector(), y.vector()),
                    _M_(p.vector(), y.vector()),
                    _S_(p.vector(), y.vector()) - eta * p.vector().inner(g),
                    -k * p.vector().inner(g),
                ]
            )
            for y, p, g in zip(Y[1:], P[:-1], G[1:])
        )

    def adjoint(self, x, Ym) -> list[df.Function]:
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )
        P = [df.Function(self.V, name="adjoint") for _ in range(len(timepoints))]
        Mdot = matrix_operator(M)
        num_intervals = timestepper.num_intervals()
        for n in tqdm.tqdm(
            range(num_intervals, 0, -1), total=num_intervals, disable=self.silent
        ):
            nj = measure_interval(n, self.td, self.timestepper)
            jump = sum(
                (
                    matmul(M, (Ym[j].vector() - self.Yd[j].vector()) / self.Yd_norms[j])
                    for j in nj
                )
            )

            solver.solve(
                P[n - 1].vector(),
                Mdot(P[n].vector()) - jump,
            )
        return P

    def dF(self, x, dx):
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        dY = cache_fetch(
            self.cache["sensitivity"],
            self.sensitivity,
            {"x": x, "dx": dx},
            x=x,
            dx=dx,
            Y=Y,
        )
        Ym = self.measure(Y)
        dYm = self.measure(dY)
        _M_ = bilinear_operator(self.model.M)
        return sum(
            [
                _M_(ym.vector() - yd.vector(), dy.vector()) / norm
                for ym, yd, dy, norm in zip(
                    Ym[1:], self.Yd[1:], dYm[1:], self.Yd_norms[1:]
                )
            ]
        )

    def hess(self, x):
        return np.array([self.hessp(x, ei) for ei in np.eye(len(x))])

    def hessp(self, x, dx):
        dt = self.timestepper.dt
        coefficients = self.coefficients.from_vector(x)
        phi = coefficients["phi"]

        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)
        dY = cache_fetch(
            self.cache["sensitivity"],
            self.sensitivity,
            {"x": x, "dx": dx},
            x=x,
            dx=dx,
            Y=Y,
        )
        dYm = self.measure(dY)

        P = cache_fetch(self.cache["adjoint"], self.adjoint, {"x": x}, x=x, Ym=Ym)
        dP = self.second_order_adjoint(x, dx, dYm, P)

        model = self.model
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"phi": phi}, phi=phi)
        _DK_ = bilinear_operator(model.DK)
        _M_ = bilinear_operator(model.M)
        _S_ = bilinear_operator(model.S)
        return dt * sum(
            np.array(
                [
                    _DK_(dp.vector(), y.vector()) + _DK_(p.vector(), dy.vector()),
                    _M_(dp.vector(), y.vector()) + _M_(p.vector(), dy.vector()),
                    _S_(dp.vector(), y.vector())
                    - dp.vector().inner(g)
                    + _S_(p.vector(), dy.vector()),
                ]
            )
            for y, dy, p, dp, g in zip(Y[1:], dY[1:], P[:-1], dP[:-1], G[1:])
        )

    def sensitivity(self, x, dx, Y) -> list[df.Function]:
        raise NotImplementedError("Need update following refactor.")
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        dD, dr, dk = dx

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()
        dY = [df.Function(self.V, name="sensitivity") for _ in range(len(timepoints))]

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )

        dL = dD * model.DK + dr * model.M + dk * model.S
        Mdot = matrix_operator(M)
        dLdot = matrix_operator(dL)
        Sdot = matrix_operator(model.S)
        G = [matmul(model.S, model.g(t)) for t in timepoints]
        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(
                dY[n + 1].vector(),
                Mdot(dY[n].vector())
                - dt * dLdot(Y[n + 1].vector())
                + dt * dk * G[n + 1],
            )
        return dY

    def second_order_adjoint(self, x, dx, dYm, P):
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        dD, dr, dk = dx

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()

        dP = [
            df.Function(self.V, name="second-order-adjoint")
            for _ in range(len(timepoints))
        ]

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        dL = dD * model.DK + dr * model.M + dk * model.S
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )

        Mdot = matrix_operator(M)
        dLdot = matrix_operator(dL)
        num_intervals = timestepper.num_intervals()
        for n in tqdm.tqdm(
            range(num_intervals, 0, -1), total=num_intervals, disable=self.silent
        ):
            nj = measure_interval(n, self.td, self.timestepper)
            jump = sum((matmul(M, dYm[j].vector()) / self.Yd_norms[j] for j in nj))
            solver.solve(
                dP[n - 1].vector(),
                Mdot(dP[n].vector()) - dt * dLdot(P[n - 1].vector()) - jump,
            )
        return dP
