#!/usr/bin/env python3
"""Generate the three Gazebo/Nav2 maps used for social-navigation collection.

The occupancy grids and Gazebo collision geometry are generated from the same
rectangle definitions.  This keeps the map, world, and pedestrian scenario
coordinates aligned and makes the dataset environments reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


RESOLUTION = 0.05
ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "gazebo_classic"
    / "hunav_gz_classic_ws"
    / "src"
    / "hunav_gazebo_wrapper"
)
MAP_DIR = WRAPPER / "maps"
WORLD_DIR = WRAPPER / "worlds"
SCENARIO_DIR = WRAPPER / "scenarios"


@dataclass(frozen=True)
class Rect:
    name: str
    cx: float
    cy: float
    sx: float
    sy: float
    height: float = 1.0
    color: tuple[float, float, float] = (0.35, 0.38, 0.40)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.sx / 2.0,
            self.cx + self.sx / 2.0,
            self.cy - self.sy / 2.0,
            self.cy + self.sy / 2.0,
        )


@dataclass(frozen=True)
class MapSpec:
    name: str
    width_m: float
    height_m: float
    free_regions: Sequence[tuple[float, float, float, float]]
    occupied: Sequence[Rect]
    start_occupied: bool
    tracks: Sequence[tuple[tuple[float, float], tuple[float, float]]]
    densities: Sequence[tuple[str, int]]


def outer_walls(width: float, height: float, thickness: float = 0.25) -> list[Rect]:
    return [
        Rect("wall_north", 0.0, height / 2.0 - thickness / 2.0, width, thickness),
        Rect("wall_south", 0.0, -height / 2.0 + thickness / 2.0, width, thickness),
        Rect("wall_west", -width / 2.0 + thickness / 2.0, 0.0, thickness, height),
        Rect("wall_east", width / 2.0 - thickness / 2.0, 0.0, thickness, height),
    ]


SPECS = [
    MapSpec(
        name="training_corridor",
        width_m=20.0,
        height_m=8.0,
        # Keep enough lateral room for two 0.7 m-diameter pedestrians and the
        # robot to pass without turning the local costmap into a solid wall.
        free_regions=[(-9.75, 9.75, -2.00, 2.00)],
        occupied=[
            Rect("corridor_wall_north", 0.0, 2.125, 20.0, 0.25),
            Rect("corridor_wall_south", 0.0, -2.125, 20.0, 0.25),
            Rect("corridor_end_west", -9.875, 0.0, 0.25, 4.50),
            Rect("corridor_end_east", 9.875, 0.0, 0.25, 4.50),
        ],
        start_occupied=True,
        tracks=[
            ((-8.0, -1.35), (8.0, -1.35)),
            ((8.0, 1.35), (-8.0, 1.35)),
            ((-7.5, -0.45), (7.5, -0.45)),
            ((7.5, 0.45), (-7.5, 0.45)),
        ],
        # Four agents are enough to create head-on and following interactions.
        # Six simultaneously spawned cyclic agents repeatedly deadlocked in the
        # same narrow section and produced long near-zero-speed trajectories.
        densities=[("low", 2), ("medium", 3), ("high", 4)],
    ),
    MapSpec(
        name="training_intersection",
        width_m=18.0,
        height_m=18.0,
        free_regions=[
            (-8.75, 8.75, -1.50, 1.50),
            (-1.50, 1.50, -8.75, 8.75),
        ],
        occupied=[
            Rect("block_north_west", -5.25, 5.25, 7.50, 7.50),
            Rect("block_north_east", 5.25, 5.25, 7.50, 7.50),
            Rect("block_south_west", -5.25, -5.25, 7.50, 7.50),
            Rect("block_south_east", 5.25, -5.25, 7.50, 7.50),
        ],
        start_occupied=True,
        tracks=[
            ((-7.8, -0.85), (7.8, -0.85)),
            ((7.8, 0.85), (-7.8, 0.85)),
            ((-0.85, -7.8), (-0.85, 7.8)),
            ((0.85, 7.8), (0.85, -7.8)),
            ((-7.2, 0.25), (7.2, 0.25)),
            ((7.2, -0.25), (-7.2, -0.25)),
            ((0.25, -7.2), (0.25, 7.2)),
            ((-0.25, 7.2), (-0.25, -7.2)),
        ],
        # Increase density progressively instead of starting the low scenario
        # with all four crossing directions active.
        densities=[("low", 2), ("medium", 4), ("high", 6)],
    ),
    MapSpec(
        name="training_doorway",
        width_m=14.0,
        height_m=12.0,
        free_regions=[],
        occupied=[
            *outer_walls(14.0, 12.0),
            # A 3.2 m opening leaves enough room for PMB2 and pedestrians to
            # pass without frequent doorway deadlocks while preserving a
            # meaningful bottleneck for social-navigation training.
            Rect("divider_north", 2.50, 3.80, 0.25, 4.40),
            Rect("divider_south", 2.50, -3.80, 0.25, 4.40),
            Rect("table_left_north", -3.8, 3.4, 1.4, 1.0, 0.85, (0.65, 0.42, 0.22)),
            Rect("table_left_south", -3.8, -3.4, 1.4, 1.0, 0.85, (0.65, 0.42, 0.22)),
            Rect("table_right_north", 5.0, 3.4, 1.2, 1.2, 0.85, (0.65, 0.42, 0.22)),
            Rect("table_right_south", 5.0, -3.4, 1.2, 1.2, 0.85, (0.65, 0.42, 0.22)),
        ],
        start_occupied=False,
        tracks=[
            ((-5.5, -0.55), (5.5, -0.55)),
            ((5.5, 0.55), (-5.5, 0.55)),
            # Additional agents start longitudinally staggered on the same
            # directional lanes instead of occupying nearly identical lateral
            # coordinates in the doorway at simulation start.
            ((-3.0, -0.55), (5.0, -0.55)),
            ((3.0, 0.55), (-5.0, 0.55)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
    ),
]


def rect_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    return (x_grid >= x0) & (x_grid <= x1) & (y_grid >= y0) & (y_grid <= y1)


def build_grid(spec: MapSpec) -> np.ndarray:
    width = round(spec.width_m / RESOLUTION)
    height = round(spec.height_m / RESOLUTION)
    xs = -spec.width_m / 2.0 + (np.arange(width) + 0.5) * RESOLUTION
    ys = spec.height_m / 2.0 - (np.arange(height) + 0.5) * RESOLUTION
    x_grid, y_grid = np.meshgrid(xs, ys)
    grid = np.zeros((height, width), dtype=np.uint8)
    if not spec.start_occupied:
        grid.fill(255)
    for region in spec.free_regions:
        grid[rect_mask(x_grid, y_grid, region)] = 255
    for obstacle in spec.occupied:
        grid[rect_mask(x_grid, y_grid, obstacle.bounds)] = 0
    return grid


def write_pgm(path: Path, grid: np.ndarray) -> None:
    header = f"P5\n# generated by scripts/generate_training_maps.py\n{grid.shape[1]} {grid.shape[0]}\n255\n"
    path.write_bytes(header.encode("ascii") + grid.tobytes())


def write_map_yaml(path: Path, spec: MapSpec) -> None:
    path.write_text(
        "\n".join(
            [
                f"image: {spec.name}.pgm",
                "mode: trinary",
                f"resolution: {RESOLUTION}",
                f"origin: [{-spec.width_m / 2.0:.3f}, {-spec.height_m / 2.0:.3f}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "",
            ]
        ),
        encoding="utf-8",
    )


def sdf_model(rect: Rect) -> str:
    r, g, b = rect.color
    return f"""
    <model name="{rect.name}">
      <static>true</static>
      <pose>{rect.cx:.3f} {rect.cy:.3f} {rect.height / 2.0:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{rect.sx:.3f} {rect.sy:.3f} {rect.height:.3f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{rect.sx:.3f} {rect.sy:.3f} {rect.height:.3f}</size></box></geometry>
          <material>
            <ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>
            <diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def write_world(path: Path, spec: MapSpec) -> None:
    models = "\n".join(sdf_model(rect) for rect in spec.occupied)
    path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    <gravity>0 0 -9.8</gravity>
    <physics name="default_physics" type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>
    <include><uri>model://sun</uri></include>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{spec.width_m:.3f} {spec.height_m:.3f} 0.02</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{spec.width_m:.3f} {spec.height_m:.3f} 0.02</size></box></geometry>
          <material>
            <ambient>0.72 0.72 0.72 1</ambient>
            <diffuse>0.72 0.72 0.72 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{models}
  </world>
</sdf>
""",
        encoding="utf-8",
    )


def scenario_text(spec: MapSpec, density: str, count: int) -> str:
    tracks = spec.tracks[:count]
    goal_lines: list[str] = []
    agent_names: list[str] = []
    agent_blocks: list[str] = []
    for index, (start, end) in enumerate(tracks, start=1):
        start_goal = 2 * index - 1
        end_goal = 2 * index
        goal_lines.extend(
            [
                f"      {start_goal}:",
                f"        x: {start[0]:.3f}",
                f"        y: {start[1]:.3f}",
                f"      {end_goal}:",
                f"        x: {end[0]:.3f}",
                f"        y: {end[1]:.3f}",
            ]
        )
        agent_names.append(f"      - agent{index}")
        heading = 0.0 if end[0] >= start[0] else 3.14159
        if abs(end[1] - start[1]) > abs(end[0] - start[0]):
            heading = 1.5708 if end[1] >= start[1] else -1.5708
        agent_blocks.append(
            f"""    agent{index}:
      id: {index}
      group_id: -1
      skin: {(index - 1) % 5}
      max_vel: {1.25 + 0.05 * ((index - 1) % 5):.2f}
      radius: 0.35
      goal_radius: 0.30
      cyclic_goals: true
      init_pose:
        x: {start[0]:.3f}
        y: {start[1]:.3f}
        z: 1.250
        h: {heading:.5f}
      behavior:
        type: Regular
        configuration: {(index - 1) % 2}
        goal_force_factor: 2.5
        obstacle_force_factor: 7.0
        social_force_factor: 3.5
        other_force_factor: 10.0
      goals:
        - {end_goal}
        - {start_goal}"""
        )
    base_name = f"agents_{spec.name}_{density}"
    return (
        "hunav_loader:\n"
        "  ros__parameters:\n"
        f"    yaml_base_name: {base_name}\n"
        "    simulator: Gazebo Classic\n"
        f"    map: {spec.name}\n"
        "    publish_people: true\n"
        "    global_goals:\n"
        + "\n".join(goal_lines)
        + "\n    agents:\n"
        + "\n".join(agent_names)
        + "\n"
        + "\n".join(agent_blocks)
        + "\n"
    )


def remove_stale_files(keep_names: Iterable[str]) -> None:
    keep = set(keep_names)
    for directory, suffixes in (
        (MAP_DIR, {".pgm", ".yaml"}),
        (WORLD_DIR, {".world"}),
        (SCENARIO_DIR, {".yaml"}),
    ):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in suffixes and path.name not in keep:
                path.unlink()


def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    keep: list[str] = []
    for spec in SPECS:
        keep.extend([f"{spec.name}.pgm", f"{spec.name}.yaml", f"{spec.name}.world"])
        for density, _ in spec.densities:
            keep.append(f"agents_{spec.name}_{density}.yaml")
    remove_stale_files(keep)

    for spec in SPECS:
        write_pgm(MAP_DIR / f"{spec.name}.pgm", build_grid(spec))
        write_map_yaml(MAP_DIR / f"{spec.name}.yaml", spec)
        write_world(WORLD_DIR / f"{spec.name}.world", spec)
        for density, count in spec.densities:
            path = SCENARIO_DIR / f"agents_{spec.name}_{density}.yaml"
            path.write_text(scenario_text(spec, density, count), encoding="utf-8")
        print(
            f"{spec.name}: {spec.width_m:.0f}x{spec.height_m:.0f} m, "
            f"densities={dict(spec.densities)}"
        )


if __name__ == "__main__":
    main()
