import numpy as np
import itertools


def schedule_next_grid(
    initial_lower_bounds,
    initial_upper_bounds,
    n_points_per_dim,
    full_evaluation_history,
):
    """
    A stateless job scheduler that determines the next grid to evaluate.
    It does not handle termination logic.

    Args:
        initial_lower_bounds (list or np.ndarray): The absolute lower bounds.
        initial_upper_bounds (list or np.ndarray): The absolute upper bounds.
        full_evaluation_history (list): A complete log of all past evaluations.
                                        If empty, the initial grid is returned.
        n_points_per_dim (int): The number of points per dimension.
        axis_transforms (list, optional): The axis transformation functions.

    Returns:
        list: A list of NumPy arrays, representing the points for the next job.
    """
    n_dims = len(initial_lower_bounds)
    initial_lower_bounds = np.array(initial_lower_bounds, dtype=float)
    initial_upper_bounds = np.array(initial_upper_bounds, dtype=float)

    # --- 1. Determine Current State from History ---

    # If history is empty, this is the first job.
    if not full_evaluation_history:
        current_lower_bounds = initial_lower_bounds
        current_upper_bounds = initial_upper_bounds
    # If history exists, determine the next bounds from the last iteration.
    else:
        grid_size = n_points_per_dim**n_dims
        last_grid_evals = full_evaluation_history[-grid_size:]
        last_grid_points = np.array([e["point"] for e in last_grid_evals])

        best_last_iter_eval = min(last_grid_evals, key=lambda x: x["value"])
        current_best_point = best_last_iter_eval["point"]

        current_lower_bounds = np.min(last_grid_points, axis=0)
        current_upper_bounds = np.max(last_grid_points, axis=0)

        # --- 2. Calculate New Bounds using Expansion/Refinement Logic ---
        new_lower_bounds = np.zeros_like(current_lower_bounds)
        new_upper_bounds = np.zeros_like(current_upper_bounds)
        all_evaluated_points = {tuple(e["point"]) for e in full_evaluation_history}

        for d in range(n_dims):
            last_axis = np.unique(last_grid_points[:, d])
            last_axis.sort()

            coord_index = np.where(np.isclose(last_axis, current_best_point[d]))[0][0]

            new_lower = last_axis[coord_index - 1] if coord_index > 0 else last_axis[0]
            new_upper = (
                last_axis[coord_index + 1]
                if coord_index < n_points_per_dim - 1
                else last_axis[-1]
            )

            # (Expansion logic remains identical to previous version)
            is_on_lower_edge = coord_index == 0
            is_on_original_lower_bound = np.isclose(
                current_lower_bounds[d], initial_lower_bounds[d]
            )
            if is_on_lower_edge and not is_on_original_lower_bound:
                candidates = [p[d] for p in all_evaluated_points if p[d] < new_lower]
                if candidates:
                    new_lower = max(candidates)

            is_on_upper_edge = coord_index == n_points_per_dim - 1
            is_on_original_upper_bound = np.isclose(
                current_upper_bounds[d], initial_upper_bounds[d]
            )
            if is_on_upper_edge and not is_on_original_upper_bound:
                candidates = [p[d] for p in all_evaluated_points if p[d] > new_upper]
                if candidates:
                    new_upper = min(candidates)

            new_lower_bounds[d] = max(new_lower, initial_lower_bounds[d])
            new_upper_bounds[d] = min(new_upper, initial_upper_bounds[d])

        current_lower_bounds = new_lower_bounds
        current_upper_bounds = new_upper_bounds

    # --- 3. Generate and Return the New Grid Points ---
    grid_axes = []
    for d in range(n_dims):
        lower_b, upper_b = current_lower_bounds[d], current_upper_bounds[d]
        final_axis = np.linspace(lower_b, upper_b, n_points_per_dim)
        grid_axes.append(final_axis)
    points_to_evaluate = list(itertools.product(*grid_axes))
    return [np.array(p) for p in points_to_evaluate]
