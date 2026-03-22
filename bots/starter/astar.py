"""A* pathfinder for 8-directional grid movement."""

from __future__ import annotations

import heapq

from cambc import Controller, Direction, Position


# 8 non-centre directions
DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]

# Delta-to-Direction lookup
_DELTA_TO_DIR: dict[tuple[int, int], Direction] = {d.delta(): d for d in DIRECTIONS}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class AStarPathfinder:
    def __init__(self, enemy_core: tuple[int, int], map_w: int, map_h: int) -> None:
        self.enemy_core = enemy_core
        self.map_w = map_w
        self.map_h = map_h
        self.known_tiles: dict[tuple[int, int], bool] = {}

    # -- grid helpers --

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_w and 0 <= y < self.map_h

    def _neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = node
        result: list[tuple[int, int]] = []
        for d in DIRECTIONS:
            dx, dy = d.delta()
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny) and self.known_tiles.get((nx, ny)) is True:
                result.append((nx, ny))
        return result

    def _is_goal(self, node: tuple[int, int]) -> bool:
        dx = node[0] - self.enemy_core[0]
        dy = node[1] - self.enemy_core[1]
        return dx * dx + dy * dy <= 2

    def _heuristic(self, node: tuple[int, int]) -> int:
        # Chebyshev distance to the nearest goal tile (any of 8 neighbors of enemy core).
        # Approximate: Chebyshev to enemy core minus 1, floored at 0.
        d = _chebyshev(node, self.enemy_core)
        return max(0, d - 1)

    # -- public interface --

    def update_grid(self, ct: Controller) -> None:
        for pos in ct.get_nearby_tiles():
            self.known_tiles[(pos.x, pos.y)] = ct.is_tile_passable(pos)

    def get_next_direction(self, start: tuple[int, int]) -> Direction | None:
        if self._is_goal(start):
            return None  # already at goal

        # A* search
        # priority queue entries: (f, tie-breaker, node, first_step)
        # first_step is the direction taken from start to reach this path
        counter = 0
        open_set: list[tuple[int, int, tuple[int, int], Direction | None]] = []
        g_score: dict[tuple[int, int], int] = {start: 0}
        closed: set[tuple[int, int]] = set()

        heapq.heappush(open_set, (self._heuristic(start), counter, start, None))

        while open_set:
            f, _, current, first_dir = heapq.heappop(open_set)

            if current in closed:
                continue
            closed.add(current)

            if self._is_goal(current):
                return first_dir  # direction of first step from start

            current_g = g_score[current]

            for nb in self._neighbors(current):
                tentative_g = current_g + 1
                if tentative_g < g_score.get(nb, float("inf")):
                    g_score[nb] = tentative_g
                    h = self._heuristic(nb)
                    counter += 1
                    # Track the first direction taken from start
                    if current == start:
                        delta = (nb[0] - start[0], nb[1] - start[1])
                        step_dir = _DELTA_TO_DIR[delta]
                    else:
                        step_dir = first_dir  # type: ignore[assignment]
                    heapq.heappush(open_set, (tentative_g + h, counter, nb, step_dir))

        return None  # no path found
