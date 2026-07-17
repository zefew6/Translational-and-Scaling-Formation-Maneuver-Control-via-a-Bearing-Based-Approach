#!/usr/bin/env python3
"""Animate formation acquisition under the paper's bearing controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bearing_formation import simulate

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class FullCanvasPillowWriter(PillowWriter):
    """Keep every GIF frame self-contained for strict image viewers."""

    def finish(self) -> None:
        full_frames = []
        for frame_index, frame in enumerate(self._frames):
            frame = frame.copy()
            # Alternating corner pixels force Pillow's delta rectangle to span
            # the complete canvas. The individual pixels are visually negligible.
            corner_color = (0, 0, 0) if frame_index % 2 else (255, 255, 255)
            width, height = frame.size
            for pixel in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
                frame.putpixel(pixel, corner_color)
            full_frames.append(frame)

        # A shared palette makes the GIF background index unambiguous. Without
        # this, some decoders fill Pillow's delta-frame margins with black.
        palette_frame = full_frames[0].convert(
            "P", palette=Image.ADAPTIVE, colors=256
        )
        paletted_frames = [palette_frame]
        paletted_frames.extend(
            frame.quantize(palette=palette_frame, dither=Image.NONE)
            for frame in full_frames[1:]
        )
        white_background = palette_frame.getpixel((0, 0))
        paletted_frames[0].save(
            self.outfile,
            save_all=True,
            append_images=paletted_frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            background=white_background,
            disposal=1,
            optimize=False,
        )


def save_animation(
    result: dict[str, object],
    output_path: Path,
    frame_count: int,
    fps: int,
) -> None:
    """Render the formation and its bearing error to an animated GIF."""
    times = result["times"]
    positions = result["positions"]
    velocities = result["velocities"]
    target_formation = result["full_target_positions"]
    bearing_error = result["bearing_error"]
    problem = result["problem"]
    leaders = problem["leaders"]
    followers = problem["followers"]
    edges = problem["edges"]

    agent_speeds = np.linalg.norm(velocities, axis=2)
    all_points = np.concatenate(
        [positions.reshape(-1, 2), target_formation.reshape(-1, 2)], axis=0
    )
    data_min = all_points.min(axis=0)
    data_max = all_points.max(axis=0)
    margin = max(1.0, 0.12 * np.max(data_max - data_min))

    fig, (ax_form, ax_speed, ax_error) = plt.subplots(
        1, 3, figsize=(14.6, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 0.9, 1.0]},
    )
    motion_label = result["metrics"]["leader_motion"].replace("-", " ")
    fig.suptitle(
        f"{positions.shape[1]} agents | {motion_label} leader velocity | "
        f"paper {result['metrics']['controller'].removeprefix('paper ')}",
        fontsize=14,
    )

    desired_edges = [
        ax_form.plot(
            [], [],
            color="0.45",
            lw=1.4,
            ls="--",
            alpha=0.7,
            label="desired formation" if edge_index == 0 else None,
            zorder=1,
        )[0]
        for edge_index, _ in enumerate(edges)
    ]
    current_edges = [
        ax_form.plot([], [], color="0.12", lw=1.5, alpha=0.85, zorder=2)[0]
        for _ in edges
    ]
    trails = []
    for i in range(positions.shape[1]):
        color = "#d1495b" if i in leaders else "#00798c"
        trails.append(ax_form.plot([], [], color=color, lw=1.8, alpha=0.55)[0])

    leader_nodes = ax_form.scatter(
        [], [], marker="^", s=85, color="#d1495b", edgecolor="white",
        linewidth=0.8, label="leaders", zorder=4,
    )
    follower_nodes = ax_form.scatter(
        [], [], marker="o", s=70, color="#00798c", edgecolor="white",
        linewidth=0.8, label="followers", zorder=4,
    )
    initial_nodes = ax_form.scatter(
        positions[0, followers, 0], positions[0, followers, 1],
        marker="x", s=55, color="#f4a261", linewidth=1.8,
        label="initial followers", zorder=3,
    )
    time_text = ax_form.text(
        0.03, 0.97, "", transform=ax_form.transAxes, va="top", fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 4},
    )
    ax_form.set(
        title="Formation maneuver",
        xlabel="x",
        ylabel="y",
        xlim=(data_min[0] - margin, data_max[0] + margin),
        ylim=(data_min[1] - margin, data_max[1] + margin),
    )
    ax_form.set_aspect("equal", adjustable="box")
    ax_form.grid(alpha=0.22)
    ax_form.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=9)

    leader_set = set(leaders.tolist())
    speed_lines = []
    for i in range(positions.shape[1]):
        is_leader = i in leader_set
        label = None
        if i == leaders[0]:
            label = "leaders"
        elif i == followers[0]:
            label = "followers"
        speed_lines.append(
            ax_speed.plot(
                [], [],
                color="#d1495b" if is_leader else "#00798c",
                lw=2.0 if is_leader else 1.4,
                alpha=0.90 if is_leader else 0.65,
                label=label,
            )[0]
        )
    max_agent_speed = max(0.05, float(np.max(agent_speeds)))
    ax_speed.set(
        title=f"All-agent speed ({motion_label})",
        xlabel="time (s)",
        ylabel=r"$\|v_i\|$",
        xlim=(0.0, float(times[-1])),
        ylim=(0.0, 1.15 * max_agent_speed),
    )
    ax_speed.grid(alpha=0.22)
    ax_speed.legend(frameon=False, loc="upper right")

    clipped_error = np.maximum(bearing_error, 1e-10)
    error_line = ax_error.plot([], [], color="#2a9d8f", lw=2.2)[0]
    error_marker = ax_error.plot([], [], "o", color="#e76f51", ms=6)[0]
    ax_error.axhline(1e-3, color="0.45", ls="--", lw=1, label=r"$10^{-3}$ threshold")
    ax_error.set_yscale("log")
    ax_error.set(
        title="Total bearing error",
        xlabel="time (s)",
        ylabel="error",
        xlim=(0.0, float(times[-1])),
        ylim=(max(1e-10, float(clipped_error.min()) / 2), float(clipped_error.max()) * 1.5),
    )
    ax_error.grid(alpha=0.22, which="both")
    ax_error.legend(frameon=False, loc="upper right")

    sampled = np.linspace(0, len(times) - 1, frame_count, dtype=int)
    frame_indices = np.concatenate(
        [np.repeat(sampled[0], 6), sampled, np.repeat(sampled[-1], 19)]
    )

    def update(k: int) -> tuple[object, ...]:
        current = positions[k]
        desired = target_formation[k]
        for desired_line, current_line, (i, j) in zip(
            desired_edges, current_edges, edges
        ):
            desired_line.set_data(
                [desired[i, 0], desired[j, 0]],
                [desired[i, 1], desired[j, 1]],
            )
            current_line.set_data(
                [current[i, 0], current[j, 0]],
                [current[i, 1], current[j, 1]],
            )
        for i, trail in enumerate(trails):
            trail.set_data(positions[: k + 1, i, 0], positions[: k + 1, i, 1])
        leader_nodes.set_offsets(current[leaders])
        follower_nodes.set_offsets(current[followers])
        error_line.set_data(times[: k + 1], clipped_error[: k + 1])
        error_marker.set_data([times[k]], [clipped_error[k]])
        for i, speed_line in enumerate(speed_lines):
            speed_line.set_data(times[: k + 1], agent_speeds[: k + 1, i])
        time_text.set_text(f"t = {times[k]:5.1f} s")
        return (
            *desired_edges,
            *current_edges,
            *trails,
            leader_nodes,
            follower_nodes,
            initial_nodes,
            error_line,
            error_marker,
            *speed_lines,
            time_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output_path,
        writer=FullCanvasPillowWriter(
            fps=fps,
            metadata={"title": "Bearing-based formation acquisition"},
        ),
        dpi=95,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--agents", type=int, default=8, help="total agent count")
    parser.add_argument(
        "--leader-motion",
        choices=("stationary", "constant", "time-varying"),
        default="stationary",
        help="prescribed leader motion; controller (7) or (10) is selected automatically",
    )
    parser.add_argument("--kp", type=float, default=1.0, help="position gain")
    parser.add_argument("--kv", type=float, default=4.0, help="velocity damping gain")
    parser.add_argument(
        "--initial-offset", type=float, default=4.0,
        help="typical follower displacement from its target",
    )
    parser.add_argument(
        "--initial-speed", type=float, default=0.0,
        help="maximum scale of randomized follower initial velocities",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--output", type=Path, default=Path("results/formation_to_target.gif")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.duration, args.dt, args.kp, args.kv, args.frames, args.fps) <= 0:
        raise ValueError("duration, dt, gains, frames, and fps must be positive.")
    if args.agents < 4 or args.initial_offset < 0 or args.initial_speed < 0:
        raise ValueError("agents >= 4; initial offset/speed must be nonnegative.")
    scenario = "forming" if args.leader_motion == "stationary" else "maneuver"
    result = simulate(
        args.duration,
        args.dt,
        args.kp,
        args.kv,
        scenario=scenario,
        n_agents=args.agents,
        initial_offset=args.initial_offset,
        initial_speed=args.initial_speed,
        seed=args.seed,
        leader_motion=args.leader_motion,
    )
    save_animation(result, args.output, args.frames, args.fps)
    metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    metrics_path.write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = result["metrics"]
    print(f"GIF: {args.output}")
    print(
        f"Agents: {metrics['number_of_agents']} "
        f"(leaders={metrics['number_of_leaders']}, "
        f"followers={metrics['number_of_followers']})"
    )
    print(
        f"Leader motion: {metrics['leader_motion']} | "
        f"controller: {metrics['controller']}"
    )
    print(
        f"Leader speed range: {metrics['minimum_leader_speed']:.6g} .. "
        f"{metrics['maximum_leader_speed']:.6g}"
    )
    print(f"Initial bearing error: {metrics['initial_total_bearing_error']:.6g}")
    print(f"Final bearing error:   {metrics['final_total_bearing_error']:.6g}")
    print(f"Final position error:  {metrics['final_position_tracking_error']:.6g}")


if __name__ == "__main__":
    main()
