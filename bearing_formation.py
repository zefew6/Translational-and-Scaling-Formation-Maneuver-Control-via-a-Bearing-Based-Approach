#!/usr/bin/env python3
"""Reproduce the constant-leader-velocity controller in Zhao and Zelazo."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np

# Keep Matplotlib usable when the user's home configuration is read-only.
_mpl_cache = Path(tempfile.gettempdir()) / "bearing-formation-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def projection_matrix(bearing: np.ndarray) -> np.ndarray:
    """Return P_g = I - g g^T for a nonzero bearing vector."""
    norm = np.linalg.norm(bearing)
    if norm <= 1e-12:
        raise ValueError("A desired edge has zero length.")
    g = bearing / norm
    return np.eye(g.size) - np.outer(g, g)


def make_problem(
    scenario: str = "maneuver",
    n_agents: int = 4,
    initial_offset: float | None = None,
    initial_speed: float = 0.0,
    seed: int = 7,
) -> dict[str, object]:
    """Create the paper example or a configurable regular-polygon formation."""
    if scenario not in {"maneuver", "forming"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    if n_agents < 4:
        raise ValueError("At least four agents are required.")
    if initial_offset is not None and initial_offset < 0:
        raise ValueError("initial_offset must be nonnegative.")
    if initial_speed < 0:
        raise ValueError("initial_speed must be nonnegative.")

    leaders = np.array([0, 1], dtype=int)
    followers = np.arange(2, n_agents, dtype=int)

    if n_agents == 4 and initial_offset is None:
        # Preserve the original Figure 2(b)-style reproduction exactly.
        reference = np.array(
            [
                [1.0, 1.0],   # leader 0: upper right
                [1.0, -1.0],  # leader 1: lower right
                [-1.0, 1.0],  # follower 2: upper left
                [-1.0, -1.0], # follower 3: lower left
            ]
        )
        edges = [(0, 1), (0, 2), (2, 3), (3, 1), (2, 1)]
        leader_positions_0 = np.array([[0.0, 5.0], [0.0, -5.0]])
        leader_velocities = np.array([[0.8, 0.18], [0.8, -0.02]])
        follower_positions_0 = np.array([[-6.0, 10.0], [-13.0, -1.0]])
        follower_velocities_0 = np.array([[-1.0, 1.5], [-1.5, -0.5]])
        effective_offset = None
    else:
        effective_offset = 4.0 if initial_offset is None else initial_offset
        angles = np.pi / n_agents - 2 * np.pi * np.arange(n_agents) / n_agents
        reference = 5.0 * np.column_stack((np.cos(angles), np.sin(angles)))

        # A polygon plus two leader fans is bearing rigid without the visual
        # clutter of a complete graph. Duplicate cycle/fan edges are removed.
        edge_set = {
            tuple(sorted((i, (i + 1) % n_agents))) for i in range(n_agents)
        }
        edge_set.update((0, j) for j in range(2, n_agents - 1))
        edge_set.update((1, j) for j in range(3, n_agents))
        edges = sorted(edge_set)

        leader_positions_0 = reference[leaders].copy()
        midpoint = np.mean(leader_positions_0, axis=0)
        leader_velocities = (
            np.array([0.18, 0.05])
            + 0.006 * (leader_positions_0 - midpoint)
        )
        rng = np.random.default_rng(seed)
        n_followers = len(followers)
        offset_angles = (
            2 * np.pi * np.arange(n_followers) / n_followers
            + 0.35
            + rng.uniform(-0.22, 0.22, n_followers)
        )
        offset_lengths = effective_offset * (
            0.65 + 0.35 * rng.random(n_followers)
        )
        offsets = offset_lengths[:, None] * np.column_stack(
            (np.cos(offset_angles), np.sin(offset_angles))
        )
        follower_positions_0 = reference[followers] + offsets

        speed_angles = offset_angles + np.pi / 2 + rng.uniform(
            -0.25, 0.25, n_followers
        )
        speed_lengths = initial_speed * (
            0.5 + 0.5 * rng.random(n_followers)
        )
        follower_velocities_0 = speed_lengths[:, None] * np.column_stack(
            (np.cos(speed_angles), np.sin(speed_angles))
        )

    if scenario == "forming":
        leader_velocities = np.zeros((2, 2))

    adjacency: list[list[int]] = [[] for _ in range(len(reference))]
    projectors: dict[tuple[int, int], np.ndarray] = {}
    desired_bearings: dict[tuple[int, int], np.ndarray] = {}
    for i, j in edges:
        delta = reference[j] - reference[i]
        g_ij = delta / np.linalg.norm(delta)
        p_ij = projection_matrix(g_ij)
        adjacency[i].append(j)
        adjacency[j].append(i)
        projectors[i, j] = p_ij
        projectors[j, i] = p_ij
        desired_bearings[i, j] = g_ij
        desired_bearings[j, i] = -g_ij

    return {
        "reference": reference,
        "edges": edges,
        "leaders": leaders,
        "followers": followers,
        "leader_positions_0": leader_positions_0,
        "leader_velocities": leader_velocities,
        "follower_positions_0": follower_positions_0,
        "follower_velocities_0": follower_velocities_0,
        "adjacency": adjacency,
        "projectors": projectors,
        "desired_bearings": desired_bearings,
        "scenario": scenario,
        "n_agents": n_agents,
        "initial_offset": effective_offset,
        "initial_speed": initial_speed,
        "seed": seed,
    }


def bearing_laplacian(
    n_agents: int,
    dimension: int,
    adjacency: list[list[int]],
    projectors: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Assemble the matrix-weighted bearing Laplacian."""
    laplacian = np.zeros((n_agents * dimension, n_agents * dimension))
    for i in range(n_agents):
        i_slice = slice(i * dimension, (i + 1) * dimension)
        for j in adjacency[i]:
            j_slice = slice(j * dimension, (j + 1) * dimension)
            p_ij = projectors[i, j]
            laplacian[i_slice, i_slice] += p_ij
            laplacian[i_slice, j_slice] -= p_ij
    return laplacian


def follower_target(
    b_ff: np.ndarray,
    b_fl: np.ndarray,
    leader_values: np.ndarray,
    dimension: int,
) -> np.ndarray:
    """Evaluate x_f* = -B_ff^{-1} B_fl x_l for positions or velocities."""
    flat = np.linalg.solve(b_ff, -b_fl @ leader_values.reshape(-1))
    return flat.reshape(-1, dimension)


def simulate(
    duration: float,
    dt: float,
    kp: float,
    kv: float,
    scenario: str = "maneuver",
    n_agents: int = 4,
    initial_offset: float | None = None,
    initial_speed: float = 0.0,
    seed: int = 7,
    leader_motion: str | None = None,
) -> dict[str, object]:
    problem = make_problem(
        scenario, n_agents, initial_offset, initial_speed, seed
    )
    reference = problem["reference"]
    edges = problem["edges"]
    leaders = problem["leaders"]
    followers = problem["followers"]
    adjacency = problem["adjacency"]
    projectors = problem["projectors"]
    desired_bearings = problem["desired_bearings"]
    leader_positions_0 = problem["leader_positions_0"]
    leader_velocities = problem["leader_velocities"]
    follower_positions_0 = problem["follower_positions_0"]
    follower_velocities_0 = problem["follower_velocities_0"]

    n_agents, dimension = reference.shape
    n_leaders = len(leaders)
    n_followers = len(followers)
    laplacian = bearing_laplacian(
        n_agents, dimension, adjacency, projectors
    )
    split = n_leaders * dimension
    b_fl = laplacian[split:, :split]
    b_ff = laplacian[split:, split:]
    b_ff_eigenvalues = np.linalg.eigvalsh(b_ff)
    if b_ff_eigenvalues[0] <= 1e-10:
        raise ValueError("B_ff is singular: the target formation is not unique.")

    if leader_motion is None:
        leader_motion = "stationary" if scenario == "forming" else "constant"
    if leader_motion not in {"stationary", "constant", "time-varying"}:
        raise ValueError(f"Unknown leader motion: {leader_motion}")
    controller = "equation (10)" if leader_motion == "time-varying" else "equation (7)"

    leader_midpoint = np.mean(leader_positions_0, axis=0)
    leader_shape = leader_positions_0 - leader_midpoint

    def leader_state(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return prescribed leader position, velocity, and acceleration."""
        if leader_motion == "stationary":
            return (
                leader_positions_0.copy(),
                np.zeros_like(leader_positions_0),
                np.zeros_like(leader_positions_0),
            )
        if leader_motion == "constant":
            return (
                leader_positions_0 + t * leader_velocities,
                leader_velocities.copy(),
                np.zeros_like(leader_positions_0),
            )

        # Sinusoidal translation and scale keep every desired bearing fixed
        # while making both leader velocities genuinely time-varying.
        translation = np.array([0.18 * t, 2.2 * np.sin(0.18 * t)])
        translation_velocity = np.array([0.18, 0.396 * np.cos(0.18 * t)])
        translation_acceleration = np.array([0.0, -0.07128 * np.sin(0.18 * t)])
        scale = 1.0 + 0.20 * np.sin(0.12 * t)
        scale_velocity = 0.024 * np.cos(0.12 * t)
        scale_acceleration = -0.00288 * np.sin(0.12 * t)
        return (
            leader_midpoint + translation + scale * leader_shape,
            translation_velocity + scale_velocity * leader_shape,
            translation_acceleration + scale_acceleration * leader_shape,
        )

    times = np.linspace(0.0, duration, round(duration / dt) + 1)
    state = np.concatenate(
        [follower_positions_0.reshape(-1), follower_velocities_0.reshape(-1)]
    )
    positions = np.zeros((times.size, n_agents, dimension))
    velocities = np.zeros_like(positions)
    leader_accelerations = np.zeros((times.size, n_leaders, dimension))

    def unpack(t: float, follower_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_f = follower_state[: n_followers * dimension].reshape(
            n_followers, dimension
        )
        v_f = follower_state[n_followers * dimension :].reshape(
            n_followers, dimension
        )
        p = np.zeros((n_agents, dimension))
        v = np.zeros_like(p)
        p_l, v_l, _ = leader_state(t)
        p[leaders] = p_l
        v[leaders] = v_l
        p[followers] = p_f
        v[followers] = v_f
        return p, v

    def dynamics(t: float, follower_state: np.ndarray) -> np.ndarray:
        p, v = unpack(t, follower_state)
        position_residual = (
            b_ff @ p[followers].reshape(-1) + b_fl @ p[leaders].reshape(-1)
        )
        velocity_residual = (
            b_ff @ v[followers].reshape(-1) + b_fl @ v[leaders].reshape(-1)
        )
        if leader_motion == "time-varying":
            _, _, a_l = leader_state(t)
            acceleration_vector = np.linalg.solve(
                b_ff,
                -b_fl @ a_l.reshape(-1)
                - kp * position_residual
                - kv * velocity_residual,
            )
        else:
            acceleration_vector = (
                -kp * position_residual - kv * velocity_residual
            )
        accelerations = acceleration_vector.reshape(n_followers, dimension)
        return np.concatenate([v[followers].reshape(-1), accelerations.reshape(-1)])

    # Fixed-step RK4 keeps this reproduction dependency-light and deterministic.
    for k, t in enumerate(times):
        positions[k], velocities[k] = unpack(t, state)
        _, _, leader_accelerations[k] = leader_state(t)
        if k == times.size - 1:
            break
        h = times[k + 1] - t
        k1 = dynamics(t, state)
        k2 = dynamics(t + h / 2, state + h * k1 / 2)
        k3 = dynamics(t + h / 2, state + h * k2 / 2)
        k4 = dynamics(t + h, state + h * k3)
        state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    target_positions = np.zeros((times.size, n_followers, dimension))
    target_velocities = np.zeros_like(target_positions)
    full_target_positions = np.zeros_like(positions)
    full_target_velocities = np.zeros_like(velocities)
    for k, t in enumerate(times):
        p_l, v_l, _ = leader_state(t)
        target_positions[k] = follower_target(b_ff, b_fl, p_l, dimension)
        target_velocities[k] = follower_target(b_ff, b_fl, v_l, dimension)
        full_target_positions[k, leaders] = p_l
        full_target_positions[k, followers] = target_positions[k]
        full_target_velocities[k, leaders] = v_l
        full_target_velocities[k, followers] = target_velocities[k]

    position_error = np.linalg.norm(
        positions[:, followers] - target_positions, axis=(1, 2)
    )
    velocity_error = np.linalg.norm(
        velocities[:, followers] - target_velocities, axis=(1, 2)
    )
    bearing_error = np.zeros(times.size)
    min_distance = np.full(times.size, np.inf)
    for k in range(times.size):
        for i, j in edges:
            delta = positions[k, j] - positions[k, i]
            bearing_error[k] += np.linalg.norm(
                delta / np.linalg.norm(delta) - desired_bearings[i, j]
            )
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                min_distance[k] = min(
                    min_distance[k], np.linalg.norm(positions[k, i] - positions[k, j])
                )

    def settling_time(signal: np.ndarray, threshold: float) -> float | None:
        above = np.flatnonzero(signal > threshold)
        if above.size == 0:
            return 0.0
        last = int(above[-1])
        return None if last == signal.size - 1 else float(times[last + 1])

    metrics = {
        "controller": f"paper {controller}",
        "scenario": scenario,
        "leader_motion": leader_motion,
        "number_of_agents": int(n_agents),
        "number_of_leaders": int(n_leaders),
        "number_of_followers": int(n_followers),
        "initial_offset": initial_offset,
        "initial_speed": float(initial_speed),
        "random_seed": int(seed),
        "duration_s": float(duration),
        "time_step_s": float(dt),
        "kp": float(kp),
        "kv": float(kv),
        "min_eigenvalue_Bff": float(b_ff_eigenvalues[0]),
        "condition_number_Bff": float(np.linalg.cond(b_ff)),
        "initial_total_bearing_error": float(bearing_error[0]),
        "final_total_bearing_error": float(bearing_error[-1]),
        "initial_position_tracking_error": float(position_error[0]),
        "final_position_tracking_error": float(position_error[-1]),
        "initial_velocity_tracking_error": float(velocity_error[0]),
        "final_velocity_tracking_error": float(velocity_error[-1]),
        "bearing_error_settling_time_at_1e-3_s": settling_time(bearing_error, 1e-3),
        "minimum_pairwise_distance": float(np.min(min_distance)),
        "minimum_leader_speed": float(
            np.min(np.linalg.norm(velocities[:, leaders], axis=2))
        ),
        "maximum_leader_speed": float(
            np.max(np.linalg.norm(velocities[:, leaders], axis=2))
        ),
    }
    return {
        "times": times,
        "positions": positions,
        "velocities": velocities,
        "target_positions": target_positions,
        "target_velocities": target_velocities,
        "full_target_positions": full_target_positions,
        "full_target_velocities": full_target_velocities,
        "leader_accelerations": leader_accelerations,
        "position_error": position_error,
        "velocity_error": velocity_error,
        "bearing_error": bearing_error,
        "problem": problem,
        "metrics": metrics,
    }


def plot_results(result: dict[str, object], output_path: Path) -> None:
    times = result["times"]
    positions = result["positions"]
    position_error = result["position_error"]
    velocity_error = result["velocity_error"]
    bearing_error = result["bearing_error"]
    problem = result["problem"]
    leaders = problem["leaders"]
    followers = problem["followers"]
    edges = problem["edges"]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    ax = axes[0]
    for i in leaders:
        ax.plot(positions[:, i, 0], positions[:, i, 1], color="#d1495b", lw=2)
    for i in followers:
        ax.plot(positions[:, i, 0], positions[:, i, 1], color="#00798c", lw=2)
    for i, j in edges:
        ax.plot(
            [positions[-1, i, 0], positions[-1, j, 0]],
            [positions[-1, i, 1], positions[-1, j, 1]],
            color="0.25",
            lw=1,
            zorder=1,
        )
    ax.scatter(
        positions[-1, leaders, 0], positions[-1, leaders, 1],
        marker="^", s=60, color="#d1495b", label="leaders", zorder=3,
    )
    ax.scatter(
        positions[-1, followers, 0], positions[-1, followers, 1],
        marker="o", s=48, color="#00798c", label="followers", zorder=3,
    )
    ax.set(title="Agent trajectories", xlabel="x", ylabel="y")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    axes[1].semilogy(times, np.maximum(bearing_error, 1e-14), color="#2a9d8f", lw=2)
    axes[1].axhline(1e-3, color="0.45", ls="--", lw=1)
    axes[1].set(title="Total bearing error", xlabel="time (s)", ylabel="error")
    axes[1].grid(alpha=0.25, which="both")

    axes[2].semilogy(
        times, np.maximum(position_error, 1e-14), color="#e76f51", lw=2,
        label="position",
    )
    axes[2].semilogy(
        times, np.maximum(velocity_error, 1e-14), color="#264653", lw=2,
        label="velocity",
    )
    axes[2].set(title="Follower tracking error", xlabel="time (s)", ylabel="2-norm")
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(frameon=False)

    fig.suptitle("Bearing-based formation maneuver control (Equation 7)", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--kp", type=float, default=0.5)
    parser.add_argument("--kv", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.dt <= 0 or args.kp <= 0 or args.kv <= 0:
        raise ValueError("duration, dt, kp, and kv must all be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = simulate(args.duration, args.dt, args.kp, args.kv)
    figure_path = args.output_dir / "core_algorithm_reproduction.png"
    metrics_path = args.output_dir / "metrics.json"
    plot_results(result, figure_path)
    metrics_path.write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"Figure:  {figure_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
