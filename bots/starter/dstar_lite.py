"""D* Lite pathfinder (Koenig & Likhachev, optimized version)."""

from __future__ import annotations

import heapq

from cambc import Controller, Direction, Position


# 8 non-centre directions
DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]

# Delta-to-Direction lookup
_DELTA_TO_DIR: dict[tuple[int, int], Direction] = {d.delta(): d for d in DIRECTIONS}

INF = float("inf")


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class DStarLitePathfinder:
    def __init__(self, enemy_core: tuple[int, int], map_w: int, map_h: int) -> None:
        self.enemy_core = enemy_core
        self.map_w = map_w
        self.map_h = map_h
        self.known_tiles: dict[tuple[int, int], bool] = {}

        # D* Lite state
        self.g: dict[tuple[int, int], float] = {}
        self.rhs: dict[tuple[int, int], float] = {}
        self.km: float = 0.0
        self.s_start: tuple[int, int] = (0, 0)  # set on first update
        self.s_last: tuple[int, int] = (0, 0)

        # Priority queue: (k1, k2, x, y) — lazy deletion via key check
        self._open: list[tuple[float, float, int, int]] = []
        self._open_set: dict[tuple[int, int], tuple[float, float]] = {}

        # Goal nodes: all in-bounds tiles adjacent to enemy core
        self.goal_nodes: set[tuple[int, int]] = set()
        ex, ey = enemy_core
        for d in DIRECTIONS:
            dx, dy = d.delta()
            nx, ny = ex + dx, ey + dy
            if 0 <= nx < map_w and 0 <= ny < map_h:
                self.goal_nodes.add((nx, ny))

        # Initialize: seed goal nodes with rhs = 0
        for goal in self.goal_nodes:
            self.rhs[goal] = 0.0
            k = self._calculate_key(goal)
            heapq.heappush(self._open, (k[0], k[1], goal[0], goal[1]))
            self._open_set[goal] = k

        self._initialized = False

    # -- helpers --

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_w and 0 <= y < self.map_h

    def _g(self, s: tuple[int, int]) -> float:
        return self.g.get(s, INF)

    def _rhs(self, s: tuple[int, int]) -> float:
        return self.rhs.get(s, INF)

    def _calculate_key(self, s: tuple[int, int]) -> tuple[float, float]:
        min_val = min(self._g(s), self._rhs(s))
        return (min_val + _chebyshev(self.s_start, s) + self.km, min_val)

    def _neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = node
        result: list[tuple[int, int]] = []
        for d in DIRECTIONS:
            dx, dy = d.delta()
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny):
                result.append((nx, ny))
        return result

    def _cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        # Cost is 1 if b is known passable, infinite otherwise
        if self.known_tiles.get(b) is True:
            return 1.0
        return INF

    def _update_vertex(self, u: tuple[int, int]) -> None:
        if u not in self.goal_nodes:
            # rhs(u) = min over neighbors s' of (c(u, s') + g(s'))
            min_rhs = INF
            for nb in self._neighbors(u):
                c = self._cost(u, nb)
                if c < INF:
                    val = c + self._g(nb)
                    if val < min_rhs:
                        min_rhs = val
            self.rhs[u] = min_rhs

        # Remove from open if present
        if u in self._open_set:
            del self._open_set[u]

        # Re-insert if inconsistent
        if self._g(u) != self._rhs(u):
            k = self._calculate_key(u)
            heapq.heappush(self._open, (k[0], k[1], u[0], u[1]))
            self._open_set[u] = k

    def _top_key(self) -> tuple[float, float]:
        # Lazy deletion: skip stale entries
        while self._open:
            k1, k2, x, y = self._open[0]
            s = (x, y)
            if s in self._open_set and self._open_set[s] == (k1, k2):
                return (k1, k2)
            heapq.heappop(self._open)
        return (INF, INF)

    def _pop(self) -> tuple[tuple[int, int], tuple[float, float]]:
        while self._open:
            k1, k2, x, y = heapq.heappop(self._open)
            s = (x, y)
            if s in self._open_set and self._open_set[s] == (k1, k2):
                del self._open_set[s]
                return (s, (k1, k2))
        return ((0, 0), (INF, INF))  # should not happen

    def _compute_shortest_path(self) -> None:
        while True:
            top = self._top_key()
            k_start = self._calculate_key(self.s_start)
            if not (top < k_start or self._rhs(self.s_start) != self._g(self.s_start)):
                break

            u, k_old = self._pop()
            k_new = self._calculate_key(u)

            if k_old < k_new:
                # Reinsert with updated key
                heapq.heappush(self._open, (k_new[0], k_new[1], u[0], u[1]))
                self._open_set[u] = k_new
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for nb in self._neighbors(u):
                    self._update_vertex(nb)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for nb in self._neighbors(u):
                    self._update_vertex(nb)

    # -- public interface --

    def update_grid(self, ct: Controller) -> None:
        pos = ct.get_position()
        current_start = (pos.x, pos.y)

        if not self._initialized:
            self.s_start = current_start
            self.s_last = current_start
            self._initialized = True

        # Scan visible tiles, detect changes
        changed: list[tuple[int, int]] = []
        for tile_pos in ct.get_nearby_tiles():
            key = (tile_pos.x, tile_pos.y)
            new_passable = ct.is_tile_passable(tile_pos)
            old_passable = self.known_tiles.get(key)
            if old_passable != new_passable:
                changed.append(key)
            self.known_tiles[key] = new_passable

        if changed:
            # Update km for start movement
            self.km += _chebyshev(self.s_last, current_start)
            self.s_last = current_start
            self.s_start = current_start

            # Update affected vertices
            for tile in changed:
                self._update_vertex(tile)
                for nb in self._neighbors(tile):
                    self._update_vertex(nb)

            self._compute_shortest_path()
        else:
            # Even if no changes, update s_start if bot moved
            if current_start != self.s_start:
                self.km += _chebyshev(self.s_last, current_start)
                self.s_last = current_start
                self.s_start = current_start
                self._compute_shortest_path()

    def get_next_direction(self, start: tuple[int, int]) -> Direction | None:
        # Check if at goal
        dx = start[0] - self.enemy_core[0]
        dy = start[1] - self.enemy_core[1]
        if dx * dx + dy * dy <= 2:
            return None

        if self._g(start) == INF:
            return None  # no path

        # Pick neighbor that minimizes c(start, s') + g(s')
        best_cost = INF
        best_dir: Direction | None = None
        for d in DIRECTIONS:
            ddx, ddy = d.delta()
            nb = (start[0] + ddx, start[1] + ddy)
            if not self._in_bounds(nb[0], nb[1]):
                continue
            c = self._cost(start, nb)
            if c < INF:
                val = c + self._g(nb)
                if val < best_cost:
                    best_cost = val
                    best_dir = d

        return best_dir
