#!/usr/bin/env python3
"""Generate the Gazebo/Nav2 maps used for social-navigation collection.

The occupancy grids and Gazebo collision geometry are generated from the same
rectangle definitions.  This keeps the map, world, and pedestrian scenario
coordinates aligned and makes the dataset environments reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


RESOLUTION = 0.05
ROOT = Path(
    os.environ.get("MOAI_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
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
    robot_waypoints: Sequence[tuple[float, float]]
    agent_radius: float = 0.33
    agent_goal_radius: float = 0.45
    agent_speed_base: float = 1.10
    agent_speed_step: float = 0.05
    goal_force_factor: float = 2.0
    obstacle_force_factor: float = 5.0
    social_force_factor: float = 2.2
    other_force_factor: float = 6.0


@dataclass(frozen=True)
class ScenarioAgent:
    start: tuple[float, float]
    end: tuple[float, float]
    speed: float
    skin: int = 0
    radius: float = 0.33
    goal_radius: float = 0.45
    waypoints: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class JointScenarioSpec:
    """A timed encounter profile tied to one collision-consistent map."""

    name: str
    map_name: str
    robot_start: tuple[float, float]
    robot_goal: tuple[float, float]
    agents: Sequence[ScenarioAgent]
    tags: Sequence[str]
    write_scenario: bool = True


def agent(start, end, speed, skin=0, waypoints=()) -> ScenarioAgent:
    return ScenarioAgent(
        start=start,
        end=end,
        speed=speed,
        skin=skin,
        waypoints=tuple(waypoints),
    )


def outer_walls(width: float, height: float, thickness: float = 0.25) -> list[Rect]:
    return [
        Rect("wall_north", 0.0, height / 2.0 - thickness / 2.0, width, thickness),
        Rect("wall_south", 0.0, -height / 2.0 + thickness / 2.0, width, thickness),
        Rect("wall_west", -width / 2.0 + thickness / 2.0, 0.0, thickness, height),
        Rect("wall_east", width / 2.0 - thickness / 2.0, 0.0, thickness, height),
    ]


SPECS = [
    MapSpec(
        name="training_oncoming_corridor",
        width_m=20.0,
        height_m=8.0,
        # A dedicated obstacle-free encounter zone isolates robot response to
        # oncoming pedestrians.  The 6 m inner width leaves room for the
        # 1.6 m teacher lane offset plus footprint and costmap clearance.
        free_regions=[(-9.75, 9.75, -3.00, 3.00)],
        occupied=[
            Rect("oncoming_wall_north", 0.0, 3.125, 20.0, 0.25),
            Rect("oncoming_wall_south", 0.0, -3.125, 20.0, 0.25),
            Rect("oncoming_end_west", -9.875, 0.0, 0.25, 6.50),
            Rect("oncoming_end_east", 9.875, 0.0, 0.25, 6.50),
        ],
        start_occupied=True,
        tracks=[((8.0, 0.0), (-8.0, 0.0))],
        densities=[("low", 1)],
        robot_waypoints=[(-8.0, 0.0), (8.0, 0.0)],
        agent_speed_base=0.70,
        agent_speed_step=0.0,
    ),
    MapSpec(
        name="training_detour_oncoming",
        width_m=24.0,
        height_m=14.0,
        free_regions=[],
        occupied=[
            *outer_walls(24.0, 14.0),
            # The obstacle is close to the robot start, leaving a long open
            # segment for a distinct oncoming encounter after the detour.
            Rect("detour_entry_block", -5.5, 0.0, 3.0, 4.0),
        ],
        start_occupied=False,
        tracks=[((9.0, 2.0), (-2.5, 2.0))],
        densities=[("low", 1)],
        robot_waypoints=[(-10.0, 1.0), (10.0, 0.0)],
        agent_radius=0.33,
        agent_goal_radius=0.45,
        agent_speed_base=0.75,
        agent_speed_step=0.0,
        goal_force_factor=2.0,
        obstacle_force_factor=5.0,
        social_force_factor=2.2,
        other_force_factor=6.0,
    ),
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
            ((-5.5, -0.45), (5.5, -0.45)),
            ((5.5, 0.45), (-5.5, 0.45)),
        ],
        # Four agents are enough to create head-on and following interactions.
        # Six simultaneously spawned cyclic agents repeatedly deadlocked in the
        # same narrow section and produced long near-zero-speed trajectories.
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-6.0, 0.0), (6.0, 0.0)],
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
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-6.0, 0.0), (6.0, 0.0)],
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
        robot_waypoints=[(-4.0, 0.0), (4.0, 0.0)],
    ),
    MapSpec(
        name="training_slalom",
        width_m=24.0,
        height_m=18.0,
        free_regions=[],
        occupied=[
            *outer_walls(24.0, 18.0),
            # Alternating walls require map-aware turns, while every bypass is
            # at least about 3 m wide.  The origin remains free for PMB2.
            Rect("slalom_upper_left", -6.0, 3.2, 1.0, 5.5),
            Rect("slalom_lower_middle", 1.8, -3.2, 1.0, 5.5),
            Rect("slalom_upper_right", 7.0, 3.2, 1.0, 5.5),
        ],
        start_occupied=False,
        # Pedestrians cross the robot's slalom route at four different x
        # positions.  This creates head-on/crossing interactions without
        # sending every agent to one shared choke point.
        tracks=[
            ((-8.2, -6.5), (-8.2, 6.5)),
            ((-2.5, 6.5), (-2.5, -6.5)),
            ((4.5, -6.5), (4.5, 6.5)),
            ((9.0, 6.5), (9.0, -6.5)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-10.0, 0.0), (10.0, 0.0)],
        agent_radius=0.33,
        agent_goal_radius=0.45,
        agent_speed_base=1.10,
        agent_speed_step=0.05,
        goal_force_factor=2.0,
        obstacle_force_factor=5.0,
        social_force_factor=2.2,
        other_force_factor=6.0,
    ),
    MapSpec(
        name="training_open_plaza",
        width_m=24.0,
        height_m=24.0,
        free_regions=[],
        occupied=[
            *outer_walls(24.0, 24.0),
            # Sparse islands preserve broad free space and create several
            # meaningful local-map contexts without narrow choke points.
            Rect("plaza_island_nw", -5.0, 5.0, 1.4, 1.4, 0.8),
            Rect("plaza_island_ne", 5.0, 5.0, 1.4, 1.4, 0.8),
            Rect("plaza_island_sw", -5.0, -5.0, 1.4, 1.4, 0.8),
            Rect("plaza_island_se", 5.0, -5.0, 1.4, 1.4, 0.8),
        ],
        start_occupied=False,
        # Four offset lanes create interactions at four different locations;
        # no single central point receives every pedestrian.
        tracks=[
            ((-9.0, -2.0), (9.0, -2.0)),
            ((9.0, 2.0), (-9.0, 2.0)),
            ((-2.0, -9.0), (-2.0, 9.0)),
            ((2.0, 9.0), (2.0, -9.0)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-9.0, 0.0), (0.0, 9.0), (9.0, 0.0), (0.0, -9.0)],
        agent_radius=0.33,
        agent_goal_radius=0.45,
        agent_speed_base=1.10,
        agent_speed_step=0.05,
        goal_force_factor=2.0,
        obstacle_force_factor=5.0,
        social_force_factor=2.2,
        other_force_factor=6.0,
    ),
    MapSpec(
        name="training_route_choice",
        width_m=22.0,
        height_m=16.0,
        free_regions=[],
        occupied=[
            *outer_walls(22.0, 16.0),
            # This is a training analogue of a route-choice problem, not a
            # copy of the held-out Dual Route map. The asymmetric obstacle is
            # centred near the map origin and gives upper/lower bypasses with
            # different geometry. Starts at y=+/-1.x make the direct line hit
            # the block while making one bypass clearly shorter.
            Rect("route_choice_block", 2.80, 0.80, 4.00, 5.40),
            Rect(
                "route_choice_upper_island",
                7.00,
                6.40,
                1.60,
                1.20,
                0.80,
                (0.55, 0.45, 0.28),
            ),
            Rect(
                "route_choice_lower_island",
                -6.80,
                -5.80,
                1.80,
                1.20,
                0.80,
                (0.55, 0.45, 0.28),
            ),
        ],
        start_occupied=False,
        # Upper/lower head-on streams expose the model to both bypasses. The
        # two vertical streams add crossings before and after the route split.
        tracks=[
            ((8.0, 5.10), (-8.0, 5.10)),
            ((8.0, -4.00), (-8.0, -4.00)),
            ((-4.2, -6.4), (-4.2, 6.4)),
            ((8.5, 6.4), (8.5, -6.4)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-8.8, -1.4), (8.8, -1.4)],
        agent_radius=0.33,
        agent_goal_radius=0.45,
        agent_speed_base=0.90,
        agent_speed_step=0.05,
        goal_force_factor=2.0,
        obstacle_force_factor=5.0,
        social_force_factor=2.2,
        other_force_factor=6.0,
    ),
    MapSpec(
        name="training_bottleneck_merge",
        width_m=22.0,
        height_m=16.0,
        free_regions=[],
        occupied=[
            *outer_walls(22.0, 16.0),
            # The long blocks create a central gate. The offset entry island
            # invalidates the direct start-goal segment, so route-aware GP
            # must choose the lower entrance before merging through the gate.
            Rect("bottleneck_upper", 2.0, 4.75, 7.0, 5.50),
            Rect("bottleneck_lower", 2.0, -4.75, 7.0, 5.50),
            Rect(
                "bottleneck_entry_island",
                -4.0,
                0.65,
                2.20,
                2.60,
                0.85,
                (0.28, 0.48, 0.34),
            ),
        ],
        start_occupied=False,
        tracks=[
            # Counterflow stays in the adjacent half of the gate and stops
            # before the entry island. A same-line head-on track made the
            # teacher collision physically unavoidable in the narrow merge.
            ((8.5, 0.45), (-2.1, 0.45)),
            ((-2.2, -6.5), (-2.2, 6.5)),
            ((6.3, 6.5), (6.3, -6.5)),
            ((-6.5, 6.5), (-6.5, -6.5)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-9.0, 0.0), (9.0, 0.0)],
        agent_speed_base=0.75,
        agent_speed_step=0.05,
    ),
    MapSpec(
        name="training_outdoor_chicane",
        width_m=26.0,
        height_m=20.0,
        free_regions=[],
        occupied=[
            *outer_walls(26.0, 20.0),
            # Staggered planter-sized blocks force an S-shaped route while
            # retaining outdoor-scale bypasses and pedestrian crossing space.
            Rect("chicane_planter_west", -3.0, 2.50, 3.0, 5.50, 0.80, (0.28, 0.48, 0.34)),
            Rect("chicane_planter_east", 3.0, -2.50, 3.0, 5.50, 0.80, (0.28, 0.48, 0.34)),
            Rect("chicane_island_nw", -9.0, 6.8, 1.8, 1.2, 0.70, (0.55, 0.45, 0.28)),
            Rect("chicane_island_se", 9.0, -6.8, 1.8, 1.2, 0.70, (0.55, 0.45, 0.28)),
        ],
        start_occupied=False,
        tracks=[
            ((0.0, -8.0), (0.0, 8.0)),
            ((-6.0, 8.0), (-6.0, -8.0)),
            ((6.0, -8.0), (6.0, 8.0)),
            ((11.0, 8.3), (-11.0, 8.3)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-11.0, 0.0), (11.0, 0.0)],
        agent_speed_base=0.75,
        agent_speed_step=0.05,
    ),
    MapSpec(
        name="training_dual_route",
        width_m=24.0,
        height_m=18.0,
        free_regions=[],
        occupied=[
            *outer_walls(24.0, 18.0),
            # Offset the block so PMB2 can spawn safely at (0, 0).  Both the
            # upper and lower routes retain more than 6 m of geometric width.
            Rect("dual_route_block", 4.5, 0.0, 5.0, 5.0),
        ],
        start_occupied=False,
        tracks=[
            ((-9.0, -4.5), (9.0, -4.5)),
            ((9.0, 4.5), (-9.0, 4.5)),
            ((-5.0, -6.5), (-5.0, 6.5)),
            ((9.0, 6.5), (9.0, -6.5)),
        ],
        densities=[("low", 2), ("medium", 3), ("high", 4)],
        robot_waypoints=[(-9.0, 0.0), (9.0, 0.0)],
        agent_radius=0.33,
        agent_goal_radius=0.45,
        agent_speed_base=1.10,
        agent_speed_step=0.05,
        goal_force_factor=2.0,
        obstacle_force_factor=5.0,
        social_force_factor=2.2,
        other_force_factor=6.0,
    ),
]


JOINT_SCENARIOS = [
    JointScenarioSpec(
        name="agents_training_oncoming_corridor_joint_head_on",
        map_name="training_oncoming_corridor",
        robot_start=(-8.0, 0.0),
        robot_goal=(8.0, 0.0),
        agents=[
            # Matching nominal speeds and symmetric starts force an encounter
            # near the map center.  With one-way coupling only the robot can
            # resolve the conflict, making avoidance causality observable.
            agent((8.0, 0.0), (-8.0, 0.0), 0.70, 1),
        ],
        tags=("oncoming", "head_on", "one_way", "moving_avoidance"),
    ),
    JointScenarioSpec(
        name="agents_training_oncoming_corridor_joint_overtaking",
        map_name="training_oncoming_corridor",
        robot_start=(-8.0, 0.0),
        robot_goal=(8.0, 0.0),
        agents=[
            # The slower pedestrian starts five metres ahead.  The robot must
            # leave the centerline, pass without stopping, then merge back.
            agent((-3.0, 0.0), (7.0, 0.0), 0.40, 2),
        ],
        tags=("overtaking", "same_direction", "one_way", "moving_avoidance"),
    ),
    JointScenarioSpec(
        name="agents_training_oncoming_corridor_joint_cut_in",
        map_name="training_oncoming_corridor",
        robot_start=(-8.0, 0.0),
        robot_goal=(8.0, 0.0),
        agents=[
            # Merge diagonally, then continue along the robot route. This
            # distinguishes a true cut-in from a momentary path crossing.
            agent(
                (-5.0, -2.2),
                (8.5, 0.0),
                0.40,
                0,
                waypoints=((-2.0, 0.0),),
            ),
        ],
        tags=(
            "cut_in",
            "diagonal_merge",
            "one_way",
            "moving_avoidance",
            "route_recovery",
        ),
    ),
    JointScenarioSpec(
        name="agents_training_oncoming_corridor_joint_staggered_two_oncoming",
        map_name="training_oncoming_corridor",
        robot_start=(-8.0, 0.0),
        robot_goal=(8.0, 0.0),
        agents=[
            # The farther pedestrian moves more slowly so the robot can finish
            # its first avoidance and recover before the second encounter.
            agent((-1.0, 0.0), (-8.0, 0.0), 0.65, 3),
            agent((8.0, 0.0), (-8.0, 0.0), 0.50, 4),
        ],
        tags=(
            "oncoming",
            "staggered",
            "two_human",
            "one_way",
            "moving_avoidance",
        ),
    ),
    JointScenarioSpec(
        name="agents_training_open_plaza_forced_crossing",
        map_name="training_open_plaza",
        robot_start=(-8.5, 0.0),
        robot_goal=(8.5, 0.0),
        agents=[
            ScenarioAgent(
                start=(0.0, -8.5),
                end=(0.0, 8.5),
                speed=0.80,
                skin=0,
                radius=0.33,
                goal_radius=0.40,
            ),
        ],
        tags=("forced_crossing", "one_way", "moving_avoidance"),
        # This scenario contains explanatory comments useful during manual
        # tuning; include it in validation/catalog generation without
        # replacing the hand-authored YAML.
        write_scenario=False,
    ),
    JointScenarioSpec(
        name="agents_training_route_choice_joint_lower_yield",
        map_name="training_route_choice",
        robot_start=(-8.2, -1.4),
        robot_goal=(9.0, -1.4),
        agents=[
            # Synchronize this crossing with the robot's lower bypass. With
            # one-way coupling the pedestrian keeps walking, so only an early
            # robot slowdown or detour can maintain social clearance.
            agent((0.0, -6.4), (0.0, 6.4), 0.35, 0),
            agent((5.8, -6.4), (5.8, 4.5), 0.55, 1),
        ],
        tags=("lower_bypass", "crossing", "yield", "static_detour"),
    ),
    JointScenarioSpec(
        name="agents_training_route_choice_joint_lower_crossing",
        map_name="training_route_choice",
        robot_start=(-8.2, -1.4),
        robot_goal=(9.0, -1.4),
        agents=[
            agent((0.0, -6.4), (0.0, 0.0), 0.55, 0),
            agent((5.8, -6.4), (5.8, -0.3), 0.55, 1),
        ],
        tags=("lower_bypass", "crossing", "static_detour"),
    ),
    JointScenarioSpec(
        name="agents_training_route_choice_joint_upper_crossing",
        map_name="training_route_choice",
        robot_start=(-8.2, 1.4),
        robot_goal=(9.0, 1.4),
        agents=[
            agent((0.0, 7.0), (0.0, 1.0), 0.45, 2),
            agent((5.5, 7.0), (5.5, 1.0), 0.45, 3),
        ],
        tags=("upper_bypass", "crossing", "static_detour"),
    ),
    JointScenarioSpec(
        name="agents_training_route_choice_joint_lower_oncoming",
        map_name="training_route_choice",
        robot_start=(-8.2, -1.4),
        robot_goal=(9.0, -1.4),
        agents=[
            # Keep the oncoming pedestrian near the obstacle so the robot's
            # safe response is biased toward the wider outer bypass.
            agent((8.7, -2.55), (-7.5, -2.55), 0.75, 1),
            agent((5.8, -6.4), (5.8, -0.3), 0.50, 4),
        ],
        tags=("lower_bypass", "oncoming", "crossing", "static_detour"),
    ),
    JointScenarioSpec(
        name="agents_training_route_choice_joint_upper_oncoming",
        map_name="training_route_choice",
        robot_start=(-8.2, 1.4),
        robot_goal=(9.0, 1.4),
        agents=[
            agent((8.7, 4.15), (-7.5, 4.15), 0.75, 2),
            agent((0.0, 7.0), (0.0, 1.0), 0.45, 0),
        ],
        tags=("upper_bypass", "oncoming", "crossing", "static_detour"),
    ),
    JointScenarioSpec(
        name="agents_training_detour_oncoming_joint_recovery",
        map_name="training_detour_oncoming",
        robot_start=(-10.0, 1.0),
        robot_goal=(10.0, 0.0),
        agents=[
            # The robot clears the entry obstacle before meeting this human in
            # the broad central segment, separating static and social actions.
            agent((9.0, 2.0), (-2.5, 2.0), 0.75, 2),
        ],
        tags=(
            "entry_detour",
            "oncoming",
            "one_way",
            "moving_avoidance",
            "route_recovery",
            "static_detour",
        ),
    ),
    JointScenarioSpec(
        name="agents_training_bottleneck_merge_joint_gate",
        map_name="training_bottleneck_merge",
        robot_start=(-9.0, 0.0),
        robot_goal=(9.0, 0.0),
        agents=[
            agent((8.5, 0.45), (-2.1, 0.45), 0.72, 0),
            agent((-2.2, -6.5), (-2.2, 6.5), 0.62, 1),
            agent((6.3, 6.5), (6.3, -6.5), 0.58, 2),
        ],
        tags=("bottleneck", "merge", "oncoming", "crossing"),
    ),
    JointScenarioSpec(
        name="agents_training_outdoor_chicane_joint_crossing",
        map_name="training_outdoor_chicane",
        robot_start=(-11.0, 0.0),
        robot_goal=(11.0, 0.0),
        agents=[
            agent((0.0, -8.0), (0.0, 8.0), 0.60, 0),
            agent((-6.0, 8.0), (-6.0, -8.0), 0.70, 1),
            agent((6.0, -8.0), (6.0, 8.0), 0.55, 2),
        ],
        tags=("outdoor", "chicane", "crossing", "static_detour"),
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


def grid_cell(spec: MapSpec, point: tuple[float, float]) -> tuple[int, int]:
    """Convert a world point to the generated top-down occupancy-grid cell."""
    x, y = point
    col = int((x + spec.width_m / 2.0) / RESOLUTION)
    row = int((spec.height_m / 2.0 - y) / RESOLUTION)
    return row, col


def point_has_clearance(
    spec: MapSpec,
    grid: np.ndarray,
    point: tuple[float, float],
    clearance: float,
) -> bool:
    row, col = grid_cell(spec, point)
    radius = int(np.ceil(clearance / RESOLUTION))
    row0, row1 = row - radius, row + radius + 1
    col0, col1 = col - radius, col + radius + 1
    if row0 < 0 or col0 < 0 or row1 > grid.shape[0] or col1 > grid.shape[1]:
        return False
    return bool(np.all(grid[row0:row1, col0:col1] == 255))


def validate_spec(spec: MapSpec) -> None:
    """Reject maps that would start actors in, or route them through, walls."""
    grid = build_grid(spec)
    if not point_has_clearance(spec, grid, (0.0, 0.0), 0.60):
        raise ValueError(f"{spec.name}: PMB2 spawn (0, 0) lacks 0.60 m clearance")

    for waypoint in spec.robot_waypoints:
        if not point_has_clearance(spec, grid, waypoint, 0.65):
            raise ValueError(
                f"{spec.name}: robot waypoint {waypoint} lacks 0.65 m clearance"
            )

    required_clearance = spec.agent_radius + 0.25
    starts: list[tuple[float, float]] = []
    for track_index, (start, end) in enumerate(spec.tracks, start=1):
        starts.append(start)
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        sample_count = max(int(np.ceil(length / 0.10)), 1)
        for alpha in np.linspace(0.0, 1.0, sample_count + 1):
            point = (
                start[0] + alpha * (end[0] - start[0]),
                start[1] + alpha * (end[1] - start[1]),
            )
            if not point_has_clearance(spec, grid, point, required_clearance):
                raise ValueError(
                    f"{spec.name}: pedestrian track {track_index} approaches an "
                    f"obstacle near ({point[0]:.2f}, {point[1]:.2f})"
                )

    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            separation = float(np.hypot(
                starts[first][0] - starts[second][0],
                starts[first][1] - starts[second][1],
            ))
            if separation < 2.0 * spec.agent_radius + 0.50:
                raise ValueError(
                    f"{spec.name}: pedestrian starts {first + 1} and {second + 1} "
                    f"are too close ({separation:.2f} m)"
                )


def validate_joint_scenario(
    scenario: JointScenarioSpec,
    specs_by_name: dict[str, MapSpec],
) -> None:
    if scenario.map_name not in specs_by_name:
        raise ValueError(f"{scenario.name}: unknown map {scenario.map_name}")
    spec = specs_by_name[scenario.map_name]
    grid = build_grid(spec)
    for label, point in (("robot start", scenario.robot_start), ("robot goal", scenario.robot_goal)):
        if not point_has_clearance(spec, grid, point, 0.65):
            raise ValueError(f"{scenario.name}: {label} {point} lacks 0.65 m clearance")

    starts: list[tuple[float, float]] = []
    for index, profile in enumerate(scenario.agents, start=1):
        starts.append(profile.start)
        route_points = (profile.start, *profile.waypoints, profile.end)
        for segment_start, segment_end in zip(route_points, route_points[1:]):
            length = float(np.hypot(
                segment_end[0] - segment_start[0],
                segment_end[1] - segment_start[1],
            ))
            sample_count = max(int(np.ceil(length / 0.10)), 1)
            for alpha in np.linspace(0.0, 1.0, sample_count + 1):
                point = (
                    segment_start[0] + alpha * (segment_end[0] - segment_start[0]),
                    segment_start[1] + alpha * (segment_end[1] - segment_start[1]),
                )
                if not point_has_clearance(
                    spec,
                    grid,
                    point,
                    profile.radius + 0.25,
                ):
                    raise ValueError(
                        f"{scenario.name}: agent {index} track approaches an "
                        f"obstacle near ({point[0]:.2f}, {point[1]:.2f})"
                    )

    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            separation = float(np.hypot(
                starts[first][0] - starts[second][0],
                starts[first][1] - starts[second][1],
            ))
            minimum = scenario.agents[first].radius + scenario.agents[second].radius + 0.50
            if separation < minimum:
                raise ValueError(
                    f"{scenario.name}: agent starts {first + 1} and {second + 1} "
                    f"are too close ({separation:.2f} m)"
                )


def joint_scenario_text(
    scenario: JointScenarioSpec,
    map_spec: MapSpec,
) -> str:
    goal_lines: list[str] = []
    agent_names: list[str] = []
    agent_blocks: list[str] = []
    next_goal_id = 1
    for index, profile in enumerate(scenario.agents, start=1):
        route_points = (profile.start, *profile.waypoints, profile.end)
        route_goal_ids: list[int] = []
        for point in route_points:
            goal_id = next_goal_id
            next_goal_id += 1
            route_goal_ids.append(goal_id)
            goal_lines.extend(
                [
                    f"      {goal_id}:",
                    f"        x: {point[0]:.3f}",
                    f"        y: {point[1]:.3f}",
                ]
            )
        agent_names.append(f"      - agent{index}")
        first_target = route_points[1]
        heading = float(np.arctan2(
            first_target[1] - profile.start[1],
            first_target[0] - profile.start[0],
        ))
        agent_goal_lines = "\n".join(
            f"        - {goal_id}" for goal_id in route_goal_ids[1:]
        )
        agent_blocks.append(
            f"""    agent{index}:
      id: {index}
      group_id: -1
      skin: {profile.skin % 5}
      max_vel: {profile.speed:.2f}
      radius: {profile.radius:.2f}
      goal_radius: {profile.goal_radius:.2f}
      cyclic_goals: false
      init_pose:
        x: {profile.start[0]:.3f}
        y: {profile.start[1]:.3f}
        z: 1.250
        h: {heading:.5f}
      behavior:
        type: Regular
        configuration: 1
        goal_force_factor: {map_spec.goal_force_factor:.1f}
        obstacle_force_factor: {map_spec.obstacle_force_factor:.1f}
        social_force_factor: {map_spec.social_force_factor:.1f}
        other_force_factor: {map_spec.other_force_factor:.1f}
      goals:
{agent_goal_lines}"""
        )
    return (
        "hunav_loader:\n"
        "  ros__parameters:\n"
        f"    yaml_base_name: {scenario.name}\n"
        "    simulator: Gazebo Classic\n"
        f"    map: {scenario.map_name}\n"
        "    publish_people: true\n"
        "    global_goals:\n"
        + "\n".join(goal_lines)
        + "\n    agents:\n"
        + "\n".join(agent_names)
        + "\n"
        + "\n".join(agent_blocks)
        + "\n"
    )


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
      max_vel: {spec.agent_speed_base + spec.agent_speed_step * ((index - 1) % 5):.2f}
      radius: {spec.agent_radius:.2f}
      goal_radius: {spec.agent_goal_radius:.2f}
      cyclic_goals: true
      init_pose:
        x: {start[0]:.3f}
        y: {start[1]:.3f}
        z: 1.250
        h: {heading:.5f}
      behavior:
        type: Regular
        # configuration=1 makes HuNav use the explicit force factors below.
        # configuration=0 silently replaces them with the much stronger
        # built-in defaults (social=5, obstacle=10), which caused abrupt
        # deceleration and in-place rotations when agents approached.
        configuration: 1
        goal_force_factor: {spec.goal_force_factor:.1f}
        obstacle_force_factor: {spec.obstacle_force_factor:.1f}
        social_force_factor: {spec.social_force_factor:.1f}
        other_force_factor: {spec.other_force_factor:.1f}
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


def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    # Keep manually authored safe-teacher scenarios and historical maps.
    # Declared specs are overwritten deterministically below, so deleting
    # unrecognised files here is both unnecessary and destructive.

    specs_by_name = {spec.name: spec for spec in SPECS}

    for spec in SPECS:
        validate_spec(spec)
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

    catalog: dict[str, dict[str, object]] = {}
    for scenario in JOINT_SCENARIOS:
        validate_joint_scenario(scenario, specs_by_name)
        scenario_file = f"{scenario.name}.yaml"
        scenario_path = SCENARIO_DIR / scenario_file
        if scenario.write_scenario:
            scenario_path.write_text(
                joint_scenario_text(scenario, specs_by_name[scenario.map_name]),
                encoding="utf-8",
            )
        elif not scenario_path.exists():
            raise FileNotFoundError(
                f"manually authored joint scenario is missing: {scenario_path}"
            )
        catalog[scenario_file] = {
            "map": scenario.map_name,
            "robot_start": list(scenario.robot_start),
            "robot_goal": list(scenario.robot_goal),
            "tags": list(scenario.tags),
            "agent_count": len(scenario.agents),
        }
        print(
            f"{scenario.name}: map={scenario.map_name}, "
            f"agents={len(scenario.agents)}, tags={','.join(scenario.tags)}"
        )
    (SCENARIO_DIR / "joint_interaction_catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
