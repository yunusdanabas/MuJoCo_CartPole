# tests/test_visualizer.py
import matplotlib
matplotlib.use("Agg")  # headless

import jax.numpy as jnp
import pytest
from lib.visualizer import plot_trajectory


def test_visualizer_output(tmp_path):
    """Test that visualizer produces expected output"""
    input_data = jnp.array([1, 2, 3])
    
    # Test that we can create a plot
    fig = plot_trajectory(input_data, input_data)
    assert fig is not None
    
    # Test that we can save the plot
    outfile = tmp_path / "test_plot.png"
    fig.savefig(outfile)
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_visualizer_edge_case(tmp_path):
    """Test visualizer with edge cases"""
    # Test with empty data
    empty_data = jnp.array([])
    try:
        fig = plot_trajectory(empty_data, empty_data)
        # If it doesn't raise an error, save and check
        if fig is not None:
            outfile = tmp_path / "empty_plot.png"
            fig.savefig(outfile)
            assert outfile.exists()
    except (ValueError, IndexError):
        # This is acceptable for empty data
        pass


def test_plot_trajectory_basic(tmp_path):
    """Test basic trajectory plotting functionality"""
    xs = jnp.linspace(0, 1, 10)
    ys = jnp.sin(2 * jnp.pi * xs)
    
    fig = plot_trajectory(xs, ys)
    outfile = tmp_path / "traj.png"
    fig.savefig(outfile)
    assert outfile.exists() and outfile.stat().st_size > 0


def test_plot_trajectory_with_different_sizes(tmp_path):
    """Test trajectory plotting with different data sizes"""
    # Small dataset
    xs_small = jnp.array([0., 1.])
    ys_small = jnp.array([0., 1.])
    fig_small = plot_trajectory(xs_small, ys_small)
    assert fig_small is not None
    
    # Larger dataset
    xs_large = jnp.linspace(0, 10, 100)
    ys_large = jnp.cos(xs_large)
    fig_large = plot_trajectory(xs_large, ys_large)
    assert fig_large is not None
    
    # Save both to ensure they work
    outfile_small = tmp_path / "small_traj.png"
    outfile_large = tmp_path / "large_traj.png"
    
    fig_small.savefig(outfile_small)
    fig_large.savefig(outfile_large)
    
    assert outfile_small.exists() and outfile_small.stat().st_size > 0
    assert outfile_large.exists() and outfile_large.stat().st_size > 0