from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


XY = Tuple[float, float]


class RollingLaserMapProvider:
    """A robot-centered occupancy map built from the latest 2-D laser scan.

    Cell values follow the SPU-BERT scene convention: 0 unknown, 1 free, 2 occupied.
    Unknown cells remain visible to the model and are treated as blocked by the
    execution safety checks by default.
    """

    def __init__(
        self,
        *,
        size_m: float = 24.0,
        resolution: float = 0.10,
        free_gap_fill_m: float = 0.20,
        unknown_is_occupied: bool = True,
    ) -> None:
        self.resolution = float(resolution)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        cells = max(8, int(math.ceil(float(size_m) / self.resolution)))
        self.width = cells
        self.height = cells
        self.size_m = cells * self.resolution
        self.free_gap_fill_m = max(float(free_gap_fill_m), 0.0)
        self.unknown_is_occupied = bool(unknown_is_occupied)
        self.origin_x = -0.5 * self.size_m
        self.origin_y = -0.5 * self.size_m
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.ready = False

    def update_from_scan(
        self,
        *,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        free_ray_limit_m: float = 11.5,
    ) -> None:
        self.origin_x = float(robot_x) - 0.5 * self.size_m
        self.origin_y = float(robot_y) - 0.5 * self.size_m
        self.grid.fill(0)
        start = self._world_to_cell(float(robot_x), float(robot_y))
        if start is None:
            self.ready = False
            return

        maximum_free_range = min(
            max(float(free_ray_limit_m), 0.0),
            max(float(range_max), 0.0),
            0.5 * self.size_m - 2.0 * self.resolution,
        )
        angle = float(angle_min)
        cyaw = math.cos(float(robot_yaw))
        syaw = math.sin(float(robot_yaw))
        for raw_range in ranges:
            value = float(raw_range)
            hit = math.isfinite(value) and float(range_min) <= value <= float(range_max)
            if hit:
                ray_range = min(value, maximum_free_range)
                hit = value <= maximum_free_range + 1e-6
            elif math.isinf(value) and value > 0.0:
                ray_range = maximum_free_range
            else:
                angle += float(angle_increment)
                continue

            local_x = ray_range * math.cos(angle)
            local_y = ray_range * math.sin(angle)
            world_x = float(robot_x) + cyaw * local_x - syaw * local_y
            world_y = float(robot_y) + syaw * local_x + cyaw * local_y
            end = self._world_to_cell(world_x, world_y)
            angle += float(angle_increment)
            if end is None:
                continue
            cells = self._bresenham(start[0], start[1], end[0], end[1])
            free_cells = cells[:-1] if hit and cells else cells
            for col, row in free_cells:
                if self.grid[row, col] != 2:
                    self.grid[row, col] = 1
            if hit and cells:
                col, row = cells[-1]
                self.grid[row, col] = 2

        self._fill_free_gaps()
        start_col, start_row = start
        self.grid[start_row, start_col] = 1
        self.ready = True

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[Tuple[int, int]]:
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                return points
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x += sx
            if doubled <= dx:
                error += dx
                y += sy

    def _fill_free_gaps(self) -> None:
        radius = int(round(self.free_gap_fill_m / self.resolution))
        if radius <= 0:
            return
        free_rows, free_cols = np.where(self.grid == 1)
        if free_rows.size == 0:
            return
        original = self.grid.copy()
        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
                if row_offset * row_offset + col_offset * col_offset > radius * radius:
                    continue
                rows = free_rows + row_offset
                cols = free_cols + col_offset
                valid = (
                    (rows >= 0)
                    & (rows < self.height)
                    & (cols >= 0)
                    & (cols < self.width)
                )
                rows = rows[valid]
                cols = cols[valid]
                unknown = original[rows, cols] == 0
                self.grid[rows[unknown], cols[unknown]] = 1
        self.grid[original == 2] = 2

    def _world_to_cell(self, x: float, y: float):
        col = int(math.floor((float(x) - self.origin_x) / self.resolution))
        row = int(math.floor((float(y) - self.origin_y) / self.resolution))
        if 0 <= col < self.width and 0 <= row < self.height:
            return col, row
        return None

    def occupancy_at_world(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = np.floor((xs - self.origin_x) / self.resolution).astype(np.int64)
        rows = np.floor((ys - self.origin_y) / self.resolution).astype(np.int64)
        valid = (
            (cols >= 0)
            & (cols < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        values = np.zeros(xs.shape, dtype=np.float32)
        values[valid] = self.grid[rows[valid], cols[valid]].astype(np.float32)
        return values

    def point_occupied_with_radius(self, x: float, y: float, radius: float) -> bool:
        center = self._world_to_cell(float(x), float(y))
        if center is None:
            return True
        radius_cells = int(math.ceil(max(float(radius), 0.0) / self.resolution))
        center_col, center_row = center
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(center_col - radius_cells, center_col + radius_cells + 1):
                if not (0 <= col < self.width and 0 <= row < self.height):
                    return True
                dx = (col - center_col) * self.resolution
                dy = (row - center_row) * self.resolution
                if math.hypot(dx, dy) > float(radius) + 0.5 * self.resolution:
                    continue
                value = int(self.grid[row, col])
                if value == 2 or (value == 0 and self.unknown_is_occupied):
                    return True
        return False

    def path_collision_cost(self, path, radius: float, weight: float) -> float:
        return sum(
            float(weight)
            for x, y in path
            if self.point_occupied_with_radius(float(x), float(y), float(radius))
        )

    def scene_patches(
        self,
        *,
        trans: XY,
        theta: float,
        env_range: float,
        env_resol: float,
        patch_size: int,
        side_patches: int,
    ):
        del env_range
        side_cells = int(side_patches) * int(patch_size)
        min_local = -0.5 * side_cells * float(env_resol)
        indices = np.arange(side_cells, dtype=np.float32)
        local_xs = min_local + indices * float(env_resol) + 0.5 * float(env_resol)
        local_ys = min_local + indices * float(env_resol) + 0.5 * float(env_resol)
        grid_x, grid_y = np.meshgrid(local_xs, local_ys)

        world_x = grid_x * math.cos(theta) - grid_y * math.sin(theta) - float(trans[0])
        world_y = grid_x * math.sin(theta) + grid_y * math.cos(theta) - float(trans[1])
        local_values = self.occupancy_at_world(world_x, world_y)

        patches = []
        attention = []
        for patch_row in range(int(side_patches)):
            for patch_col in range(int(side_patches)):
                patch = local_values[
                    patch_row * patch_size : (patch_row + 1) * patch_size,
                    patch_col * patch_size : (patch_col + 1) * patch_size,
                ]
                patches.append(patch.astype(np.float32).reshape(-1))
                attention.append(1.0 if np.any(patch > 0.0) else 0.0)
        return np.stack(patches, axis=0), np.asarray(attention, dtype=np.float32)

    def occupancy_grid_data(self) -> list[int]:
        output = np.full(self.grid.shape, -1, dtype=np.int8)
        output[self.grid == 1] = 0
        output[self.grid == 2] = 100
        return output.reshape(-1).tolist()
