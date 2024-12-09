import matplotlib.pyplot as plt
import numpy as np


def plot_point_cloud(points, save_path, title="2D Point Cloud",
                     figsize=(10, 8), point_size=20, point_color='blue',
                     point_alpha=0.6):
    """
    Create and save a scatter plot of a 2D point cloud.

    Parameters:
    -----------
    points : array-like
        Input points with shape (N, 2) where N is the number of points
    save_path : str
        Path where the plot should be saved
    title : str, optional
        Title of the plot (default: "2D Point Cloud")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 8))
    point_size : float, optional
        Size of the scatter points (default: 20)
    point_color : str, optional
        Color of the scatter points (default: 'blue')
    point_alpha : float, optional
        Transparency of points, between 0 and 1 (default: 0.6)

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If points array doesn't have shape (N, 2)
    """
    # Convert input to numpy array if it isn't already
    points = np.array(points)

    # Input validation
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must have shape (N, 2)")

    # Create new figure
    plt.figure(figsize=figsize)

    # Create scatter plot
    plt.scatter(points[:, 0], points[:, 1],
                s=point_size,
                c=point_color,
                alpha=point_alpha)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Make plot neat
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        plt.close()
        raise Exception(f"Error saving plot: {str(e)}")


def plot_multi_quiver(source_points, arrows, save_path,
                      title="Multi-Vector Quiver Plot",
                      figsize=(10, 8),
                      arrow_scale=1.0,
                      arrow_width=0.005,
                      arrow_colors=None,
                      arrow_alpha=0.6):
    """
    Create and save a quiver plot with multiple vectors at each source point.

    Parameters:
    -----------
    source_points : array-like
        Source points with shape (N, 2) where N is the number of points
    arrows : array-like
        Vectors with shape (N, M, 2) where M is the number of vectors per point
        Each vector is represented by [dx, dy]
    save_path : str
        Path where the plot should be saved
    title : str, optional
        Title of the plot (default: "Multi-Vector Quiver Plot")
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 8))
    arrow_scale : float, optional
        Scaling factor for arrow size (default: 1.0)
    arrow_width : float, optional
        Width of the arrows (default: 0.005)
    arrow_colors : list or None, optional
        List of M colors for different vector types (default: None)
    arrow_alpha : float, optional
        Transparency of arrows, between 0 and 1 (default: 0.6)

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If input arrays don't have correct shapes or are inconsistent
    """
    # Convert inputs to numpy arrays
    source_points = np.array(source_points)
    arrows = np.array(arrows)

    # Input validation
    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError("source_points must have shape (N, 2)")
    if arrows.ndim != 3 or arrows.shape[2] != 2:
        raise ValueError("arrows must have shape (N, M, 2)")
    if arrows.shape[0] != source_points.shape[0]:
        raise ValueError("Number of source points and arrow sets must match")

    # Get dimensions
    N, M, _ = arrows.shape

    # Set default colors if none provided
    if arrow_colors is None:
        arrow_colors = plt.cm.rainbow(np.linspace(0, 1, M))
    elif len(arrow_colors) != M:
        raise ValueError("Number of colors must match number of vectors per point")

    # Create new figure
    plt.figure(figsize=figsize)

    # Plot source points
    plt.scatter(source_points[:, 0], source_points[:, 1],
                color='black', alpha=0.5, label='Source Points')

    # Plot arrows for each vector type
    for m in range(M):
        # Extract vectors for current type
        vectors = arrows[:, m, :]

        # Create quiver plot
        plt.quiver(source_points[:, 0], source_points[:, 1],
                   vectors[:, 0], vectors[:, 1],
                   angles='xy', scale_units='xy', scale=arrow_scale,
                   width=arrow_width,
                   color=arrow_colors[m],
                   alpha=arrow_alpha,
                   label=f'Vector Type {m + 1}')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Make plot neat and adjust for legend
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        plt.close()
        raise Exception(f"Error saving plot: {str(e)}")
