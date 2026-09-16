import dolfin as df
import numpy as np
import tqdm

from glymphopt.assigners import SuperspaceAssigner, VolumetricConcentration
from glymphopt.cache import CacheObject, cache_fetch
from glymphopt.measure import LossFunction
from glymphopt.operators import (
    bilinear_operator,
    boundary_mass_matrices,
    diffusion_operator_matrices,
    mass_matrices,
    matmul,
    matrix_operator,
    transfer_matrix,
)
from glymphopt.timestepper import TimeStepper


class TwocompartmentModel:
    def __init__(self, W, D, g):
        self.W = W
        self.Me, self.Mp = mass_matrices(W)
        self.DKe, self.DKp = diffusion_operator_matrices(D, W)
        self.Se, self.Sp = boundary_mass_matrices(W)
        self.T = transfer_matrix(W)
        self.g = g

    def M(self, coefficients) -> df.Matrix:
        n_e = coefficients["n_e"]
        n_p = coefficients["n_p"]
        return n_e * self.Me + n_p * self.Mp

    def L(self, coefficients) -> df.Matrix:
        n_e = coefficients["n_e"]
        n_p = coefficients["n_p"]
        t_ep = coefficients["t_ep"]
        t_pb = coefficients["t_pb"]
        k_e = coefficients["k_e"]
        k_p = coefficients["k_p"]
        gamma = coefficients["gamma"]
        Mp = self.Mp
        DKe = self.DKe
        DKp = self.DKp
        T = self.T
        Se = self.Se
        Sp = self.Sp
        return (
            (n_e * DKe + gamma * n_p * DKp)
            + t_ep * T
            + t_pb * Mp
            + (k_e * Se + k_p * Sp)
        )

    def preconditioner(self, coefficients) -> None | df.Matrix:
        return None


class MulticompartmentInverseProblem:
    def __init__(
        self,
        td,
        Yd,
        coefficientvector,
        g,
        D,
        dt=3600,
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
        self.W = df.FunctionSpace(
            self.V.mesh(), as_vector_element(self.V.ufl_element(), dim=2)
        )

        self.model = TwocompartmentModel(self.W, D=D, g=g)
        self.cache = {
            "state": CacheObject(),
            "adjoint": CacheObject(),
            "g": CacheObject(),
            "sensitivity": CacheObject(),
            "operator": CacheObject(),
        }

        self.coefficients = coefficientvector
        self.superassigner = SuperspaceAssigner(self.V, self.W)

        coefficients = coefficientvector.coefficients
        n_e, n_p = coefficients["n_e"], coefficients["n_p"]
        self.volumetric_concentration = VolumetricConcentration(
            (n_e, n_p), self.V, self.W
        )
        self.loss = LossFunction(self.td, self.Yd)

    def F(self, x):
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)
        return self.loss(Ym)

    def measure(self, Y, step=False):
        Ym = [df.Function(self.V, name="measured_state") for _ in range(len(self.td))]

        if step:  # Necessary for implicit Euler adjoint.
            find_intervals = self.timestepper.find_intervals(self.td)
            for i, _ in enumerate(self.td[1:], start=1):
                ni = find_intervals[i]
                Ym[i].assign(self.volumetric_concentration(Y[ni + 1]))
            return Ym

        Y_before = [df.Function(self.V, name="before") for _ in range(len(self.td))]
        Y_after = [df.Function(self.V, name="after") for _ in range(len(self.td))]
        find_intervals = self.timestepper.find_intervals(self.td)
        dt = self.timestepper.dt
        time = self.timestepper.vector()
        for i, ti in enumerate(self.td[1:], start=1):
            ni = find_intervals[i]
            Y_before[i].assign(self.volumetric_concentration(Y[ni]))
            Y_after[i].assign(self.volumetric_concentration(Y[ni + 1]))
            step_fraction = (ti - time[ni]) / dt
            Ym[i].assign(
                ((1 - step_fraction) * Y_before[i] + step_fraction * Y_after[i])
            )
        return Ym

    def forward(self, x):
        coefficients = self.coefficients.from_vector(x)
        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()

        Y = [df.Function(self.W, name="state") for _ in range(len(timepoints))]
        Y[0].assign(self.superassigner(self.Yd[0]))

        model = self.model
        M = model.M(coefficients)
        L = model.L(coefficients)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"x": x}, x=x)
        solver = cache_fetch(
            self.cache["operator"],
            df.KrylovSolver,
            {"x": x},
            A=M + dt * L,
            method="cg",
            preconditioner="jacobi",
        )
        Mdot = matrix_operator(M)
        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(Y[n + 1].vector(), Mdot(Y[n].vector()) + dt * G[n + 1])
        return Y

    def boundary_vectors(self, x):
        model = self.model
        timestepper = self.timestepper
        coefficients = self.coefficients.from_vector(x)
        k_e = coefficients["k_e"]
        k_p = coefficients["k_p"]
        eta = coefficients["eta"]
        phi = coefficients["n_e"] + coefficients["n_p"]
        S = (eta * k_e / phi) * model.Se + (eta * k_p / phi) * model.Sp
        return [
            matmul(S, self.superassigner(model.g.update(t)).vector())
            for t in timestepper.vector()
        ]

    def gradF(self, x):
        return NotImplementedError("Missing adjoint definition for twocompartment")
        coefficients = self.coefficients.from_vector(x)
        phi = coefficients["phi"]
        dt = self.timestepper.dt
        model = self.model
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)

        P = cache_fetch(self.cache["adjoint"], self.adjoint, {"x": x}, x=x, Ym=Ym)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"phi": phi}, phi=phi)

        _M_ = bilinear_operator(model.M)
        _DK_ = bilinear_operator(model.DK)
        _S_ = bilinear_operator(model.S)
        return dt * sum(
            np.array(
                [
                    _DK_(p.vector(), y.vector()),
                    _M_(p.vector(), y.vector()),
                    _S_(p.vector(), y.vector()) - p.vector().inner(g),
                ]
            )
            for y, p, g in zip(Y[1:], P[:-1], G[1:])
        )


def as_vector_element(el, dim):
    return df.VectorElement(
        family=el.family(), cell=el.cell(), degree=el.degree(), dim=dim
    )
