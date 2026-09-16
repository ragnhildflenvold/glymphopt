import json

import click
import numpy as np
import dolfin as df
import tqdm
import scipy

from dolfin import inner, grad, dot


from glymphopt.cache import CacheObject, cache_fetch
from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.datageneration import BoundaryConcentration

from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.measure import measure
from glymphopt.operators import (
    matrix_operator,
    bilinear_operator,
    matmul,
)
from glymphopt.parameters import default_twocomp_parameters, singlecomp_parameters
from glymphopt.scale import create_reduced_problem
from glymphopt.timestepper import TimeStepper


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
def main(input, output):
    coefficients = singlecomp_parameters(default_twocomp_parameters())
    coeffconverter = CoefficientVector(coefficients, ("a", "r", "k", "eta"))

    domain = read_mesh(input)
    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry = read_function_data(input, domain, "boundary_concentration")
    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)

    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    problem = InverseProblem(input, coeffconverter, g=g, D=D, progress=True)
    results = iterative_subproblem_optimization(problem, coeffconverter)
    with open(output, "w") as f:
        f.write(json.dumps(results, indent=2))


def print_callback(intermediate_result):
    print("-" * 60)
    print(intermediate_result)


def direct_scaled_minimizer(problem):
    a_ref = 1.0
    r_ref = 1e-6
    k_ref = 1e-2
    eta_ref = 1.0

    a0 = 1.0
    r0 = 0.0
    k0 = 1e-2
    eta0 = 0.4

    scaled_problem = create_reduced_problem(
        problem, np.array([a_ref, r_ref, k_ref, eta_ref]), [0, 1, 2, 3]
    )
    scaled_bounds = scipy.optimize.Bounds(
        [0.1, 0.0, 0.0, 0.05], [10.0, 100.0, 100.0, 1.0]
    )
    y0 = np.array([a0, r0, k0, eta0])
    scaled_problem.problem.silent = True
    scaled_solution = scipy.optimize.minimize(
        scaled_problem.F,
        x0=y0,
        method="L-BFGS-B",
        jac=scaled_problem.gradF,
        bounds=scaled_bounds,
        callback=print_callback,
    )
    print(scaled_solution)
    results_dict = {
        "fun": scaled_solution.fun,
        "x": scaled_problem.transform(scaled_solution.x).tolist(),
        "success": scaled_solution.success,
    }
    return results_dict


def iterative_subproblem_optimization(problem, coeffconverter):
    print("Solving eta only minimization problem.")
    a_ref = 1.0
    r_ref = 1e-6
    k_ref = 1e-2
    eta_ref = 1.0

    # After scaling
    a0 = 1.0
    r0 = 0.0
    k0 = 1e-1
    eta0 = 0.4

    eta_problem = create_reduced_problem(problem, np.array([a0, r0, k0, eta_ref]), [3])
    eta_bounds = scipy.optimize.Bounds([0.05], [1.0])
    y0 = np.array([eta0])
    eta_problem.problem.silent = True
    sol_eta = scipy.optimize.minimize(
        eta_problem.F,
        x0=y0,
        method="L-BFGS-B",
        jac=eta_problem.gradF,
        bounds=eta_bounds,
        callback=print_callback,
    )
    print(sol_eta)
    print(f"Min x: {sol_eta.x}")
    print("-" * 60)
    diffusion_problem = create_reduced_problem(
        problem, np.array([a_ref, r0, k0, eta_ref]), [0, 3]
    )
    diffusion_bounds = scipy.optimize.Bounds([0.1, 0.05], [10.0, 1.0])
    y0 = np.array([a0, *sol_eta.x])
    diffusion_problem.problem.silent = True
    sol_diffusion = scipy.optimize.minimize(
        diffusion_problem.F,
        x0=y0,
        method="L-BFGS-B",
        jac=diffusion_problem.gradF,
        hess=diffusion_problem.hess,
        bounds=diffusion_bounds,
        callback=print_callback,
    )
    print(sol_diffusion)
    print(sol_diffusion.x)

    print("-" * 60)
    diffusion_conductivity_problem = create_reduced_problem(
        problem, np.array([a_ref, r0, k_ref, eta_ref]), [0, 2, 3]
    )
    diffusion_conductivity_bounds = scipy.optimize.Bounds(
        [0.1, 0.0, 0.05], [10.0, 100.0, 1.0]
    )
    y0 = np.array([sol_diffusion.x[0], k0, sol_diffusion.x[1]])
    diffusion_conductivity_problem.problem.silent = True
    sol_diffusion_conductivity = scipy.optimize.minimize(
        diffusion_conductivity_problem.F,
        x0=y0,
        method="L-BFGS-B",
        jac=diffusion_conductivity_problem.gradF,
        hess=diffusion_conductivity_problem.hess,
        bounds=diffusion_conductivity_bounds,
        callback=print_callback,
    )
    print(sol_diffusion_conductivity)

    scaled_problem = create_reduced_problem(
        problem, np.array([a_ref, r_ref, k_ref, eta_ref]), [0, 1, 2, 3]
    )
    scaled_bounds = scipy.optimize.Bounds(
        [0.1, 0.0, 0.0, 0.05], [10.0, 100.0, 100.0, 1.0]
    )
    y0 = np.array(
        [
            sol_diffusion_conductivity.x[0],  # a
            r0,
            sol_diffusion_conductivity.x[1],  # k
            sol_diffusion_conductivity.x[2],  # eta
        ]
    )
    scaled_problem.problem.silent = True
    scaled_solution = scipy.optimize.minimize(
        scaled_problem.F,
        x0=y0,
        method="L-BFGS-B",
        jac=scaled_problem.gradF,
        bounds=scaled_bounds,
        callback=print_callback,
    )
    print(scaled_solution)
    results_dict = {
        "fun": scaled_solution.fun,
        "x": scaled_problem.transform(scaled_solution.x).tolist(),
        "success": scaled_solution.success,
    }
    return results_dict


class Model:
    def __init__(self, V, D=None, g=None):
        D = D or df.Identity(V.mesh().topology().dim())

        domain = V.mesh()
        dx = df.Measure("dx", domain)
        ds = df.Measure("ds", domain)

        u, v = df.TrialFunction(V), df.TestFunction(V)
        self.M = df.assemble(inner(u, v) * dx)
        self.DK = df.assemble(inner(dot(D, grad(u)), grad(v)) * dx)
        self.S = df.assemble(inner(u, v) * ds)
        self.g = g or BoundaryConcentration(V)


def gradient_sensitivities(F, x, **kwargs):
    return np.array([F(x, ei, **kwargs) for ei in np.eye(len(x))])


def measure_interval(n: int, td: np.ndarray, timestepper: TimeStepper):
    bins = np.digitize(td, timestepper.vector(), right=True)
    return list(np.where(n == bins)[0])


class InverseProblem:
    def __init__(
        self,
        data_path,
        coefficientvector,
        dt=3600,
        timescale=1.0,
        g=None,
        D=None,
        progress=True,
    ):
        self.silent = not progress
        domain = read_mesh(data_path)
        self.td, self.Yd = read_function_data(data_path, domain, "concentration")

        t_start = self.td[0]
        dt = dt * timescale
        N = int(np.ceil(np.round((self.td[-1] - t_start) / dt, 12)))
        t_end = N * dt
        self.timestepper = TimeStepper(dt, (t_start, t_end))

        coefficients = coefficientvector.coefficients

        self.V = self.Yd[0].function_space()
        g = g or BoundaryConcentration(self.V, timescale * 3600)
        D = D or coefficients["D"] * df.Identity(domain.topology().dim())
        self.model = Model(self.V, g=g, D=D)

        self.cache = {
            "state": CacheObject(),
            "adjoint": CacheObject(),
            "g": CacheObject(),
            "sensitivity": CacheObject(),
            "operator": CacheObject(),
        }

        _M_ = bilinear_operator(self.model.M)
        self.Yd_norms = [_M_(yi.vector(), yi.vector()) for yi in self.Yd]
        self.coefficients = coefficientvector

    def F(self, x):
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = measure(self.timestepper, Y, self.td)
        _M_ = bilinear_operator(self.model.M)
        timepoint_errors = [
            _M_(Ym_i.vector() - Yd_i.vector(), Ym_i.vector() - Yd_i.vector()) / norm_i
            for Ym_i, Yd_i, norm_i in zip(Ym[1:], self.Yd[1:], self.Yd_norms[1:])
        ]
        J = 0.5 * sum(timepoint_errors)
        return J

    def forward(self, x):
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        eta = coefficients["eta"]

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()
        Y = [df.Function(self.V, name="state") for _ in range(len(timepoints))]
        Y[0].assign(self.Yd[0])

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"eta": eta}, eta=eta)
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )
        Mdot = matrix_operator(M)

        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(Y[n + 1].vector(), Mdot(Y[n].vector()) + dt * k * G[n + 1])
        return Y

    def boundary_vectors(self, eta):
        model = self.model
        timestepper = self.timestepper
        return [eta * matmul(model.S, model.g(t)) for t in timestepper.vector()]

    def gradF(self, x):
        coefficients = self.coefficients.from_vector(x)
        k = coefficients["k"]
        eta = coefficients["eta"]
        dt = self.timestepper.dt
        model = self.model
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = measure(self.timestepper, Y, self.td)

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
        timestepper = self.timestepper

        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        dY = cache_fetch(
            self.cache["sensitivity"],
            self.sensitivity,
            {"x": x, "dx": dx},
            x=x,
            dx=dx,
            Y=Y,
        )
        Ym = measure(timestepper, Y, self.td)
        dYm = measure(timestepper, dY, self.td)
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
        Ym = measure(self.timestepper, Y, self.td)
        dY = cache_fetch(
            self.cache["sensitivity"],
            self.sensitivity,
            {"x": x, "dx": dx},
            x=x,
            dx=dx,
            Y=Y,
        )
        dYm = measure(self.timestepper, dY, self.td)

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


if __name__ == "__main__":
    main()
