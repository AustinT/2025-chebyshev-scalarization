import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def create_grid(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a grid of points in [0,1] x [0,1]."""
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.flatten(), Y.flatten()]).T
    return X, Y, points, x

def is_feasible_convex(points: np.ndarray) -> np.ndarray:
    """Unit circle centered at (0,0) cropped to [0,1]x[0,1]."""
    return points[:, 0]**2 + points[:, 1]**2 <= 1

def is_feasible_concave(points: np.ndarray) -> np.ndarray:
    """Outside circle of radius 0.9 centered at (1,1), cropped to [0,1]x[0,1]."""
    return ((points[:, 0] - 1)**2 + (points[:, 1] - 1)**2 >= 0.9**2)

def is_feasible_mixed(points: np.ndarray) -> np.ndarray:
    """Convex region with a concave semi-circular slice taken out of it.
    Excludes points within a radius of 0.1 of the point (0.75, 0.75)."""
    # Convex condition: unit circle centered at (0,0)
    convex_condition = points[:, 0]**2 + points[:, 1]**2 <= 1
    
    # Exclusion condition: another circle in top-right
    exclusion_condition = (points[:, 0] - 1.1)**2 + (points[:, 1] - 1.1)**2 < 0.7**2
    
    return convex_condition & ~exclusion_condition

def linear_scalarization(points: np.ndarray, w1: float, z1: float, z2: float) -> np.ndarray:
    """Linear scalarization: w1 * (x1 - z1) + (1-w1) * (x2 - z2)."""
    return w1 * (points[:, 0] - z1) + (1 - w1) * (points[:, 1] - z2)

def chebyshev_scalarization(points: np.ndarray, w1: float, z1: float, z2: float) -> np.ndarray:
    """Chebyshev scalarization: -max(w1 * (z1 - x1), (1-w1) * (z2 - x2))."""
    return -np.maximum(w1 * (z1 - points[:, 0]), (1 - w1) * (z2 - points[:, 1]))

def augmented_chebyshev(points: np.ndarray, w1: float, z1: float, z2: float, rho: float) -> np.ndarray:
    """Augmented Chebyshev: chebyshev + rho * linear."""
    cheby = chebyshev_scalarization(points, w1, z1, z2)
    linear = linear_scalarization(points, w1, z1, z2)
    return cheby + rho * linear

def find_optimal_points(points: np.ndarray, feasible: np.ndarray, 
                        scalarization_values: np.ndarray) -> np.ndarray:
    """Find points that maximize the scalarization function in the feasible region."""
    feasible_values = scalarization_values.copy()
    feasible_values[~feasible] = -np.inf
    max_indices = np.where(feasible_values == np.max(feasible_values))[0]
    return points[max_indices]

def create_visualization(n_points: int = 100, n_slider_steps: int = 50, 
                        z1: float = 1.1, z2: float = 1.1, rho: float = 0.01):
    """Create the interactive visualization with all components."""
    # Create grid
    X, Y, points, x_range = create_grid(n_points)
    
    # Define feasibility functions
    feasibility_functions = {
        'convex': is_feasible_convex,
        'concave': is_feasible_concave,
        'mixed': is_feasible_mixed
    }
    
    # Define scalarization functions
    scalarizations = {
        'linear': lambda p, w1: linear_scalarization(p, w1, z1, z2),
        'chebyshev': lambda p, w1: chebyshev_scalarization(p, w1, z1, z2),
        'augmented_chebyshev': lambda p, w1: augmented_chebyshev(p, w1, z1, z2, rho)
    }
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Linear", "Chebyshev", "Augmented Chebyshev"]
    )
    
    # W1 slider values
    w1_values = np.linspace(0, 1, n_slider_steps)
    
    # Create frames for animation
    frames = []
    
    for w1 in w1_values:
        frame_data = []
        
        for i, (scalar_name, scalar_func) in enumerate(scalarizations.items()):
            col = i + 1
            
            for feasibility_name, feasibility_func in feasibility_functions.items():
                # Calculate feasibility
                feasible = feasibility_func(points)
                
                # Calculate scalarization values
                scalar_values = scalar_func(points, w1)
                
                # Reshape for contour
                Z_scalar = scalar_values.reshape(X.shape)
                Z_feasible = feasible.reshape(X.shape)
                
                # Find optimal points
                optimal_points = find_optimal_points(points, feasible, scalar_values)
                
                # Create trace objects for frame_data (without explicit visibility)
                contour_obj = go.Contour(
                    z=Z_scalar,
                    x=x_range,
                    y=x_range,
                    colorscale='Viridis',
                    showscale=False,
                    name=f"{scalar_name}_contour_{feasibility_name}",
                    contours=dict(
                        coloring='lines',
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    )
                )
                
                feasible_region_obj = go.Heatmap(
                    z=Z_feasible.astype(float),
                    x=x_range,
                    y=x_range,
                    colorscale=[[0, 'rgba(255, 0, 0, 0.3)'], [1, 'rgba(0, 255, 0, 0.3)']],
                    showscale=False,
                    name=f"{scalar_name}_feasible_{feasibility_name}"
                )
                
                optimal_trace_obj = go.Scatter(
                    x=optimal_points[:, 0],
                    y=optimal_points[:, 1],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name=f"{scalar_name}_optimal_{feasibility_name}"
                )
                
                frame_data.extend([contour_obj, feasible_region_obj, optimal_trace_obj])
                
                # Add traces to the figure for the initial frame (w1=w1_values[0])
                # with specific visibility settings.
                if w1 == w1_values[0]:
                    is_initially_visible = (feasibility_name == 'convex')
                    
                    # Contour for initial figure state
                    initial_contour = go.Contour(**contour_obj.to_plotly_json())
                    initial_contour.visible = is_initially_visible
                    fig.add_trace(initial_contour, row=1, col=col)
                    
                    # Feasible region for initial figure state
                    initial_feasible_region = go.Heatmap(**feasible_region_obj.to_plotly_json())
                    initial_feasible_region.visible = is_initially_visible
                    fig.add_trace(initial_feasible_region, row=1, col=col)
                    
                    # Optimal points for initial figure state
                    initial_optimal_trace = go.Scatter(**optimal_trace_obj.to_plotly_json())
                    initial_optimal_trace.visible = is_initially_visible
                    fig.add_trace(initial_optimal_trace, row=1, col=col)
        
        frames.append(go.Frame(data=frame_data, name=f"w1={w1:.2f}"))
    
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
                            transition=dict(duration=0)
                        )
                    ],
                    label=f"{w1:.2f}"
                )
                for w1 in w1_values
            ]
        )
    ]
    
    # Create dropdown for feasible region selection
    dropdown_buttons = []
    for feasibility_name in feasibility_functions.keys():
        visible_traces = []
        for i in range(len(fig.data)):
            if feasibility_name in fig.data[i].name:
                visible_traces.append(True)
            else:
                visible_traces.append(False)
                
        dropdown_buttons.append(
            dict(
                method="update",
                args=[{"visible": visible_traces}],
                label=feasibility_name.capitalize()
            )
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        title="Multi-Objective Optimization Scalarization Methods",
        sliders=sliders,
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                type="dropdown",
                name="Feasible Region"
            )
        ]
    )
    
    # Update axes properties
    for i in range(1, 4):
        fig.update_xaxes(title_text="x₁", range=[0, 1], row=1, col=i)
        fig.update_yaxes(title_text="x₂", range=[0, 1], row=1, col=i)
    
    return fig

def main(standalone: bool = False,
         n_points: int = 100, n_slider_steps: int = 50, 
         z1: float = 1.1, z2: float = 1.1, rho: float = 0.05,
         output_file: str = "multi_objective_scalarization.html"):
    """Generate the visualization and save it to an HTML file."""
    fig = create_visualization(n_points, n_slider_steps, z1, z2, rho)
    html_kwargs = dict()
    if not standalone:
        html_kwargs = dict(include_plotlyjs='/assets/js/plotly-3.0.1.min.js')
    pio.write_html(fig, file=output_file, auto_open=True, auto_play=False, **html_kwargs)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", action="store_true")
    args = parser.parse_args()
    main(standalone=not args.website)
