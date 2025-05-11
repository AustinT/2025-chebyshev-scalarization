import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def create_grid(
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a grid of points in [0,1] x [0,1]."""
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.flatten(), Y.flatten()]).T
    return X, Y, points, x


def is_feasible_convex(points: np.ndarray) -> np.ndarray:
    """Unit circle centered at (0,0) cropped to [0,1]x[0,1]."""
    return points[:, 0] ** 2 + points[:, 1] ** 2 <= 1


def is_feasible_concave(points: np.ndarray) -> np.ndarray:
    """Outside circle of radius 0.9 centered at (1,1), cropped to [0,1]x[0,1]."""
    return (points[:, 0] - 1) ** 2 + (points[:, 1] - 1) ** 2 >= 0.9**2


def is_feasible_mixed(points: np.ndarray) -> np.ndarray:
    """Convex region with a concave semi-circular slice taken out of it.
    Excludes points within a radius of 0.1 of the point (0.75, 0.75)."""
    # Convex condition: unit circle centered at (0,0)
    convex_condition = points[:, 0] ** 2 + points[:, 1] ** 2 <= 1

    # Exclusion condition: another l_p circle in top-right
    p = 6
    exclusion_condition = (points[:, 0] - 1.1) ** p + (points[:, 1] - 1.1) ** p < 0.7**p

    return convex_condition & ~exclusion_condition


def linear_scalarization(
    points: np.ndarray, w1: float, z1: float, z2: float
) -> np.ndarray:
    """Linear scalarization: w1 * (x1 - z1) + (1-w1) * (x2 - z2)."""
    return w1 * (points[:, 0] - z1) + (1 - w1) * (points[:, 1] - z2)


def chebyshev_scalarization(
    points: np.ndarray, w1: float, z1: float, z2: float
) -> np.ndarray:
    """Chebyshev scalarization: -max(w1 * (z1 - x1), (1-w1) * (z2 - x2))."""
    return -np.maximum(w1 * (z1 - points[:, 0]), (1 - w1) * (z2 - points[:, 1]))


def augmented_chebyshev(
    points: np.ndarray, w1: float, z1: float, z2: float, rho: float
) -> np.ndarray:
    """Augmented Chebyshev: chebyshev + rho * linear."""
    cheby = chebyshev_scalarization(points, w1, z1, z2)
    linear = linear_scalarization(points, w1, z1, z2)
    return cheby + rho * linear


def find_optimal_points(
    points: np.ndarray, feasible: np.ndarray, scalarization_values: np.ndarray
) -> np.ndarray:
    """Find points that maximize the scalarization function in the feasible region."""
    feasible_values = scalarization_values.copy()
    feasible_values[~feasible] = -np.inf
    max_indices = np.where(feasible_values == np.max(feasible_values))[0]
    return points[max_indices]


def create_visualization(
    n_points: int = 100,
    n_slider_steps: int = 50,
    z1: float = 1.1,
    z2: float = 1.1,
    rho: float = 0.01,
):
    """Create the interactive visualization with all components."""
    # Create grid
    X, Y, points, x_range = create_grid(n_points)

    # Define feasibility functions
    feasibility_functions = {
        "convex": is_feasible_convex,
        "concave": is_feasible_concave,
        "mixed": is_feasible_mixed,
    }

    # Define scalarization functions
    scalarizations = {
        "linear": lambda p, w1: linear_scalarization(p, w1, z1, z2),
        "chebyshev": lambda p, w1: chebyshev_scalarization(p, w1, z1, z2),
        "augmented_chebyshev": lambda p, w1: augmented_chebyshev(p, w1, z1, z2, rho),
    }

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=["Linear", "Chebyshev", "Augmented Chebyshev"]
    )

    # W1 slider values
    eps = 1e-3
    w1_values = np.linspace(eps, 1 - eps, n_slider_steps)

    # Initialize figure with all traces (for w1_values[0])
    # and identify indices of traces that will be animated
    initial_w1 = w1_values[0]
    animated_trace_indices = []
    trace_counter = 0

    for i, (scalar_name, scalar_func) in enumerate(scalarizations.items()):
        col = i + 1
        for feasibility_name, feasibility_func in feasibility_functions.items():
            # Calculate data for initial_w1
            feasible = feasibility_func(points)
            scalar_values_initial = scalar_func(points, initial_w1)
            optimal_points_initial = find_optimal_points(
                points, feasible, scalar_values_initial
            )

            Z_scalar_initial = scalar_values_initial.reshape(X.shape)
            Z_feasible_static = feasible.reshape(
                X.shape
            )  # Static for this feasibility_name

            is_visible = feasibility_name == "convex"  # Initial visibility

            # 1. Contour Trace (animated)
            contour_obj_initial = go.Contour(
                z=Z_scalar_initial,
                x=x_range,
                y=x_range,
                colorscale="Viridis",
                showscale=False,
                name=f"{scalar_name}_contour_{feasibility_name}",
                contours=dict(
                    coloring="lines",
                    showlabels=True,
                    labelfont=dict(size=12, color="white"),
                ),
                visible=is_visible,
            )
            fig.add_trace(contour_obj_initial, row=1, col=col)
            animated_trace_indices.append(trace_counter)
            trace_counter += 1

            # 2. Feasible Region Trace (static with w1)
            feasible_region_obj_static = go.Heatmap(
                z=Z_feasible_static.astype(float),
                x=x_range,
                y=x_range,
                colorscale=[[0, "rgba(211, 211, 211, 0.5)"], [1, "rgba(70, 130, 180, 0.5)"]],
                showscale=False,
                name=f"{scalar_name}_feasible_{feasibility_name}",
                visible=is_visible,
            )
            fig.add_trace(feasible_region_obj_static, row=1, col=col)
            # Not added to animated_trace_indices
            trace_counter += 1

            # 3. Optimal Points Trace (animated)
            optimal_trace_obj_initial = go.Scatter(
                x=optimal_points_initial[:, 0],
                y=optimal_points_initial[:, 1],
                mode="markers",
                marker=dict(color="black", size=10),
                name=f"{scalar_name}_optimal_{feasibility_name}",
                visible=is_visible,
            )
            fig.add_trace(optimal_trace_obj_initial, row=1, col=col)
            animated_trace_indices.append(trace_counter)
            trace_counter += 1

    # Create frames for animation
    frames = []
    for w1_frame_val in w1_values:
        frame_data_for_animation = (
            []
        )  # Will contain 18 trace objects for animated traces

        # Iterate in the same order as initial trace creation to match animated_trace_indices
        for scalar_name_frame, scalar_func_frame in scalarizations.items():
            for (
                feasibility_name_frame,
                feasibility_func_frame,
            ) in feasibility_functions.items():
                # Calculate data for current w1_frame_val
                current_feasible_def = feasibility_func_frame(points)
                current_scalar_values = scalar_func_frame(points, w1_frame_val)
                current_optimal_points = find_optimal_points(
                    points, current_feasible_def, current_scalar_values
                )

                # Create Contour object for the frame
                contour_obj_frame = go.Contour(
                    z=current_scalar_values.reshape(X.shape),
                    x=x_range,
                    y=x_range,
                    colorscale="Viridis",
                    showscale=False,
                    name=f"{scalar_name_frame}_contour_{feasibility_name_frame}",  # Name for consistency
                    contours=dict(
                        coloring="lines",
                        showlabels=True,
                        labelfont=dict(size=12, color="white"),
                    ),
                )
                frame_data_for_animation.append(contour_obj_frame)

                # Feasible region is NOT added here (it's static)

                # Create Optimal Points object for the frame
                optimal_trace_obj_frame = go.Scatter(
                    x=current_optimal_points[:, 0],
                    y=current_optimal_points[:, 1],
                    mode="markers",
                    marker=dict(color="black", size=10),
                    name=f"{scalar_name_frame}_optimal_{feasibility_name_frame}",  # Name for consistency
                )
                frame_data_for_animation.append(optimal_trace_obj_frame)

        frames.append(
            go.Frame(
                data=frame_data_for_animation,
                name=f"w1={w1_frame_val:.2f}",
                traces=animated_trace_indices,
            )
        )

    # Add frames to the figure
    fig.frames = frames

    # Create slider
    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "w₁: "},
            pad={"t": 50},
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"w1={w1:.2f}"],
                        dict(
                            frame=dict(duration=100, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f"{w1:.2f}",
                )
                for w1 in w1_values
            ],
        )
    ]

    # Create dropdown for feasible region selection
    dropdown_buttons = []
    for feasibility_name_dropdown in feasibility_functions.keys():
        visible_traces_flags = []  # List of 27 booleans for fig.data
        for trace_in_fig in fig.data:  # fig.data has 27 traces
            # trace_in_fig.name is like "linear_contour_convex"
            if feasibility_name_dropdown in trace_in_fig.name:
                visible_traces_flags.append(True)
            else:
                visible_traces_flags.append(False)

        dropdown_buttons.append(
            dict(
                method="update",
                args=[{"visible": visible_traces_flags}],
                label=feasibility_name_dropdown.capitalize(),
            )
        )

    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        title=None,
        sliders=sliders,
        showlegend=False,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.20,
                yanchor="top",
                type="dropdown",
                name="Feasible Region",
            )
        ],
    )

    # Update axes properties
    for i in range(1, 4):
        fig.update_xaxes(title_text="f₁", range=[0, 1], row=1, col=i)
        fig.update_yaxes(title_text="f₂", range=[0, 1], row=1, col=i)

    return fig


def main(
    standalone: bool = True,
    n_points: int = 72,
    n_slider_steps: int = 25,
    z1: float = 1.1,
    z2: float = 1.1,
    rho: float = 0.05,
    output_file: str = "multi_objective_scalarization.html",
):
    """Generate the visualization and save it to an HTML file."""
    fig = create_visualization(n_points, n_slider_steps, z1, z2, rho)
    html_kwargs = dict()
    if not standalone:
        html_kwargs = dict(include_plotlyjs="/assets/js/plotly-3.0.1.min.js")
    pio.write_html(
        fig, file=output_file, auto_open=True, auto_play=False, **html_kwargs
    )
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", action="store_true")
    args = parser.parse_args()
    main(standalone=not args.website)
