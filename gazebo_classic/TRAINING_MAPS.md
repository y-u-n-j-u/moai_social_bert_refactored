# SPU-BERT training maps

The dataset collection suite contains three controlled social-navigation maps.
Their occupancy grids and Gazebo collision models are generated together by:

```bash
./scripts/generate_training_maps.py
```

| Map | Size | Main interaction | Pedestrian scenarios |
|---|---:|---|---|
| `training_corridor` | 20 x 8 m | frontal approach, passing, following | 2 / 4 / 6 |
| `training_intersection` | 18 x 18 m | perpendicular crossing | 4 / 6 / 8 |
| `training_doorway` | 14 x 12 m | 1.5 m doorway bottleneck | 2 / 4 / 6 |

Each map has `low`, `medium`, and `high` density scenario YAML files. The
terminal menu discovers these files automatically.

The model receives an 8 x 8 m robot-centered map downsampled to 32 x 32 cells.
The controlled maps place the relevant wall, intersection, or doorway inside
that local field of view during the interaction. The global maps remain large
enough to provide 8--20 m robot routes for guidance-point training.

## Basic waypoint curriculum

Automatic collection defaults to repeatable centerline waypoint routes:

| Map | Robot route |
|---|---|
| `training_corridor` | `(-8, 0) <-> (8, 0)` |
| `training_doorway` | `(-5.5, 0) <-> (5.5, 0)` |
| `training_intersection` | `(-7.5, 0) <-> (7.5, 0)` |

Pedestrian timing, direction, speed, and density provide variation while the
robot first learns basic passing, yielding, following, doorway, and crossing
interactions. Set `HUNAV_AUTO_GOAL_MODE=random` only for a later
generalization-collection stage. A custom route can be supplied as
`HUNAV_AUTO_GOAL_WAYPOINTS='-8,0;8,0'`.

## Collection policy

- Collect all three density levels rather than training on one fixed count.
- Use automatic waypoints only after Nav2 validates `ComputePathToPose`.
- Keep complete goal episodes together when creating train/val/test splits.
- Hold out complete runs, and preferably one layout variant, for testing.
- Reject or label timeout, collision, and long stationary episodes.
- Record the map name, scenario YAML, random seed, and Nav2 result with each
  recording before scaling collection.

## Files

```text
hunav_gazebo_wrapper/maps/training_*.pgm
hunav_gazebo_wrapper/maps/training_*.yaml
hunav_gazebo_wrapper/worlds/training_*.world
hunav_gazebo_wrapper/scenarios/agents_training_*_{low,medium,high}.yaml
```
