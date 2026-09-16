import dolfin as df
import numpy as np
import scipy as sp

from glymphopt.timestepper import TimeStepper
from glymphopt.operators import mass_matrix, bilinear_operator, matrix_operator
from glymphopt.mri_loss import numpy_array_to_dolfin_vector, sparse_matrix_to_dolfin


def measure_interval(n: int, td: np.ndarray, timestepper: TimeStepper):
    bins = np.digitize(td, timestepper.vector(), right=True)
    return list(np.where(n == bins)[0])


def measure(
    timesteps: TimeStepper,
    states: list[df.Function],
    measure_times: list[float] | np.ndarray,
):
    V = states[0].function_space()
    Y = [df.Function(V, name="measured_state") for _ in range(len(measure_times))]
    find_intervals = timesteps.find_intervals(measure_times)
    dt = timesteps.dt
    time = timesteps.vector()
    for i, ti in enumerate(measure_times[1:], start=1):
        ni = find_intervals[i]

        # Use stepwise solution, as it is not necessarily straight forward to use
        # the linear interpolant between timepoints.
        # print(i, ni, time[ni], ti, time[ni + 1])
        # Y[i].assign(states[ni + 1])

        # Need to rederive this expression if I want to use linear interpolant
        step_fraction = (ti - time[ni]) / dt
        Y[i].assign(((1 - step_fraction) * states[ni] + step_fraction * states[ni + 1]))
    return Y


class LossFunction:
    def __init__(self, td, Yd):
        self.td = td
        self.Yd = Yd
        self.V = Yd[0].function_space()
        self.M = mass_matrix(self.V)
        self._M_ = bilinear_operator(self.M)
        self.norms = [self._M_(yi.vector(), yi.vector()) for yi in Yd]

    def __call__(self, Ym):
        timepoint_errors = [
            self._M_(Ym_i.vector() - Yd_i.vector(), Ym_i.vector() - Yd_i.vector())
            / norm_i
            for Ym_i, Yd_i, norm_i in zip(Ym[1:], self.Yd[1:], self.norms[1:])
        ]
        return 0.5 * sum(timepoint_errors)


class MRILoss:
    def __init__(self, evaluation_data):
        npzfile = np.load(evaluation_data)
        M = sp.sparse.csr_matrix(
            (
                npzfile["matrix_data"],
                npzfile["matrix_indices"],
                npzfile["matrix_indptr"],
            ),
            shape=npzfile["matrix_shape"],
        )
        Cd = [npzfile[f"vector{idx}"] for idx in range(5)]

        self.M = sparse_matrix_to_dolfin(M)
        self.Cd = [numpy_array_to_dolfin_vector(cdi) for cdi in Cd[1:]]
        self.M_ = [matrix_operator(self.M) for _ in range(4)]
        self.norm = [ci.dot(ci) for ci in Cd[1:]]

    def __call__(self, Ym):
        errs = (
            self.M_[idx](ym.vector()) - self.Cd[idx] for idx, ym in enumerate(Ym[1:])
        )
        return 0.5 * sum(e.inner(e) / norm for e, norm in zip(errs, self.norm))

    def measure(self, timesteps, Y, td, measure_op):
        V = Y[0].function_space()
        Y = [df.Function(V, name="measured_state") for _ in range(len(td))]
        find_intervals = timesteps.find_intervals(td)
        for i, _ in enumerate(td[1:], start=1):
            ni = find_intervals[i]
            Y[i].assign(measure_op(Y[ni + 1]))
        return Y
