# SPU-BERT training maps

The Gazebo dataset suite contains eight training maps and one held-out test
map. Occupancy grids, Gazebo collision models, and pedestrian scenarios are
generated from the same geometry definitions:

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored
/home/junwoo/miniconda3/envs/spubert/bin/python3.8 scripts/generate_training_maps.py
```

| Map | Size | Primary interaction | Role |
|---|---:|---|---|
| `training_corridor` | 20 x 8 m | passing and following | train |
| `training_intersection` | 18 x 18 m | perpendicular crossing | train |
| `training_doorway` | 14 x 12 m | doorway yielding | train |
| `training_slalom` | 24 x 18 m | repeated static detours | train |
| `training_open_plaza` | 24 x 24 m | unconstrained multi-human motion | train |
| `training_route_choice` | 22 x 16 m | upper/lower bypass choice | train |
| `training_bottleneck_merge` | 22 x 16 m | offset entry and narrow merge | train candidate |
| `training_outdoor_chicane` | 26 x 20 m | outdoor-style S detour | train candidate |
| `training_dual_route` | 24 x 18 m | unseen route-choice topology | held-out test only |

Every map has `low`, `medium`, and `high` density YAMLs containing 2, 3, and
4 pedestrians. The two candidate maps become admitted training sources only
after a ten-seed pilot passes the quality gate. Recordings from
`training_dual_route` must never enter train or validation PKLs.

## Route-aware input

The processed model input uses a 20 x 20 m robot-centered occupancy crop
downsampled to 32 x 32 cells. Guidance is not the point eight metres along the
straight robot-to-goal segment. It is selected by:

1. computing a collision-free Nav2/A* global path;
2. finding the path point nearest the current robot pose;
3. accumulating arc length along that path;
4. selecting the point at 8 m, or the final goal when less than 8 m remains.

The route-choice, bottleneck, and chicane designs all block the direct
start-goal segment. This makes route-aware guidance observable in the dataset
instead of allowing the model to succeed on straight open-space examples.

## Joint scenarios

Six deterministic scenarios combine static detours with purposeful pedestrian
encounters:

| Scenario suffix | Map | Agents | Interaction |
|---|---|---:|---|
| `lower_crossing` | `training_route_choice` | 2 | lower bypass + crossings |
| `upper_crossing` | `training_route_choice` | 2 | upper bypass + crossings |
| `lower_oncoming` | `training_route_choice` | 2 | lower bypass + counterflow |
| `upper_oncoming` | `training_route_choice` | 2 | upper bypass + counterflow |
| `bottleneck_merge_joint_gate` | `training_bottleneck_merge` | 3 | gate merge + adjacent counterflow |
| `outdoor_chicane_joint_crossing` | `training_outdoor_chicane` | 3 | S detour + three crossings |

Joint-scenario pedestrians use one-pass goals. A cyclic return caused an
unintended second encounter and physical overlap, so repeated timing diversity
is produced by episode start profiles and seeds instead of mid-episode U-turns.

## Design visualization

```bash
MPLCONFIGDIR=/tmp/mpl python3 scripts/visualize_joint_interaction_designs.py
```

The output is
`figures/training_map_design/joint_interaction_scenarios.png`. It overlays the
old straight direction, old GP, collision-free route, route-aware GP, and all
pedestrian tracks.

## Files

```text
hunav_gazebo_wrapper/maps/training_*.pgm
hunav_gazebo_wrapper/maps/training_*.yaml
hunav_gazebo_wrapper/worlds/training_*.world
hunav_gazebo_wrapper/scenarios/agents_training_*_{low,medium,high}.yaml
hunav_gazebo_wrapper/scenarios/agents_training_*_joint_*.yaml
hunav_gazebo_wrapper/scenarios/joint_interaction_catalog.json
```
