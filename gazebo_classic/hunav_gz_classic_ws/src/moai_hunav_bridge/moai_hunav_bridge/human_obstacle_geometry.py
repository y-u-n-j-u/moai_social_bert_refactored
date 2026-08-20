from __future__ import annotations

from math import ceil, cos, pi, sin


def filled_disc_offsets(
    radius: float,
    point_spacing: float,
    minimum_ring_points: int,
) -> list[tuple[float, float]]:
    """Return polar samples that cover a circular obstacle without hollow bands."""
    radius = max(float(radius), 0.0)
    if radius == 0.0 or int(minimum_ring_points) <= 0:
        return [(0.0, 0.0)]

    spacing = max(float(point_spacing), 0.05)
    radial_layers = max(int(ceil(radius / spacing)), 1)
    offsets = [(0.0, 0.0)]

    for layer in range(1, radial_layers + 1):
        ring_radius = radius * layer / radial_layers
        ring_points = max(
            int(minimum_ring_points),
            int(ceil(2.0 * pi * ring_radius / spacing)),
        )
        for index in range(ring_points):
            angle = 2.0 * pi * index / ring_points
            offsets.append((ring_radius * cos(angle), ring_radius * sin(angle)))

    return offsets
