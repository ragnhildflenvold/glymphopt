import numpy as np

from .grid_scheduler import GridScheduler


def adaptive_grid_search(
    func,
    lower_bounds,
    upper_bounds,
    axis_transforms=None,
    n_points_per_dim=5,
    n_iterations=4,
):
    """
    Performs an adaptive grid search using a GridScheduler.
    """
    scheduler = GridScheduler(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_points_per_dim=n_points_per_dim,
        axis_transforms=axis_transforms,
    )

    iteration_history = []
    full_evaluation_history = []
    overall_best = {"point": None, "value": float("inf")}

    print("--- Starting Adaptive Grid Search (Refactored) ---")

    for i in range(n_iterations):
        print(f"\nIteration {i + 1}/{n_iterations}")
        print("  Current Search Bounds:")
        for dim in range(scheduler.n_dims):
            is_log = (
                " (Log Scale)"
                if scheduler.axis_transforms and scheduler.axis_transforms[dim]
                else ""
            )
            print(
                f"    Dim {dim + 1}{is_log}: [{scheduler.current_lower_bounds[dim]:.4g}, {scheduler.current_upper_bounds[dim]:.4g}]"
            )

        # 1. Create Grid (delegated to scheduler)
        points_to_evaluate = scheduler.create_grid(full_evaluation_history)

        # 2. Evaluate Points & Log Full History
        values = []
        for point in points_to_evaluate:
            value = func(point)
            values.append(value)
            full_evaluation_history.append({"point": point, "value": value})
        values = np.array(values)

        min_index = np.argmin(values)
        current_best_point = points_to_evaluate[min_index]
        current_best_value = values[min_index]

        print(f"  Best point in this iteration: {np.round(current_best_point, 4)}")
        print(f"  Minimal value in this iteration: {current_best_value:.4f}")

        iteration_history.append(
            {
                "iteration": i + 1,
                "best_point": current_best_point,
                "best_value": current_best_value,
            }
        )
        if current_best_value < overall_best["value"]:
            overall_best = {"point": current_best_point, "value": current_best_value}

    print("\n--- Adaptive Grid Search Finished ---")
    print(f"Final Optimal Point: {np.round(overall_best['point'], 6)}")
    print(f"Final Optimal Value: {overall_best['value']:.6f}")

    return overall_best, iteration_history, full_evaluation_history
