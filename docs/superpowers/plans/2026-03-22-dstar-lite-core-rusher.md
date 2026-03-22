# D* Lite Core-Rusher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BUILDER_BOT random-walk with D* Lite and A* pathfinding toward the enemy core, self-destructing on arrival.

**Architecture:** Three files — `main.py` (orchestration, base class, goal inference), `astar.py` (A* pathfinder), `dstar_lite.py` (D* Lite pathfinder). A `USE_DSTAR_LITE` flag in `main.py` selects the algorithm. Both pathfinders share a `Pathfinder` base class with a common grid model.

**Tech Stack:** Python 3.12, cambc game framework (Rust engine with Python bindings)

**Note:** No unit tests — the Controller is a Rust-injected runtime object that cannot be instantiated outside a game match. Verification is done by running matches via `cambc run`.

**Spec:** `docs/superpowers/specs/2026-03-22-dstar-lite-core-rusher-design.md`

---

## File Structure

```
bots/starter/
  main.py        — (modify) Player class, goal inference, Pathfinder base class, bot orchestration
  astar.py       — (create) AStarPathfinder
  dstar_lite.py  — (create) DStarLitePathfinder
```

---

### Task 1: Create A* Pathfinder

**Files:**
- Create: `bots/starter/astar.py`

- [ ] **Step 1: Write AStarPathfinder class**

This is the simpler pathfinder — implement it first so we can verify the bot orchestration works before tackling D* Lite.

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add bots/starter/astar.py
git commit -m "feat: add A* pathfinder for builder bot"
```

---

### Task 2: Create D* Lite Pathfinder

**Files:**
- Create: `bots/starter/dstar_lite.py`

- [ ] **Step 1: Write DStarLitePathfinder class**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add bots/starter/dstar_lite.py
git commit -m "feat: add D* Lite pathfinder for builder bot"
```

---

### Task 3: Rewrite main.py with Goal Inference and Bot Orchestration

**Files:**
- Modify: `bots/starter/main.py`

- [ ] **Step 1: Rewrite main.py**

Replace the entire file with:

```python
"""Core-rusher bot — pathfinds toward the enemy core and self-destructs.

Core: spawns up to 3 builder bots on random adjacent tiles (unchanged).
Builder bot: uses D* Lite (or A*) to navigate passable tiles toward the
enemy core, then self-destructs when adjacent.
"""

import random

from cambc import Controller, Direction, EntityType, Position

from astar import AStarPathfinder
from dstar_lite import DStarLitePathfinder

# --- Configuration ---
USE_DSTAR_LITE = True

# Non-centre directions
DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]


def _infer_enemy_core(
    core_pos: Position, map_w: int, map_h: int
) -> tuple[int, int]:
    """Infer the enemy core position using map symmetry."""
    cx, cy = core_pos.x, core_pos.y
    candidates = [
        (cx, map_h - 1 - cy),          # horizontal
        (map_w - 1 - cx, cy),          # vertical
        (map_w - 1 - cx, map_h - 1 - cy),  # rotational
    ]

    # Pick the candidate furthest from our core (maximally opposite).
    # Tie-break: prefer rotational (last in list, most common).
    best = candidates[0]
    best_dist = (best[0] - cx) ** 2 + (best[1] - cy) ** 2
    for c in candidates[1:]:
        d = (c[0] - cx) ** 2 + (c[1] - cy) ** 2
        if d >= best_dist:
            best = c
            best_dist = d

    return best


def _find_allied_core(ct: Controller) -> Position | None:
    """Find allied core position by scanning nearby entities."""
    for eid in ct.get_nearby_entities():
        try:
            if ct.get_entity_type(eid) == EntityType.CORE:
                return ct.get_position(eid)
        except Exception:
            continue
    return None


class Player:
    def __init__(self) -> None:
        self.num_spawned = 0
        # Builder bot state
        self.pathfinder = None
        self.enemy_core: tuple[int, int] | None = None

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()

        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    def _run_core(self, ct: Controller) -> None:
        if self.num_spawned < 3:
            spawn_pos = ct.get_position().add(random.choice(DIRECTIONS))
            if ct.can_spawn(spawn_pos):
                ct.spawn_builder(spawn_pos)
                self.num_spawned += 1

    def _run_builder(self, ct: Controller) -> None:
        pos = ct.get_position()

        # First run: discover core, infer enemy, create pathfinder
        if self.pathfinder is None:
            core_pos = _find_allied_core(ct)
            if core_pos is None:
                return  # can't find core yet, skip turn

            map_w = ct.get_map_width()
            map_h = ct.get_map_height()
            self.enemy_core = _infer_enemy_core(core_pos, map_w, map_h)

            if USE_DSTAR_LITE:
                self.pathfinder = DStarLitePathfinder(self.enemy_core, map_w, map_h)
            else:
                self.pathfinder = AStarPathfinder(self.enemy_core, map_w, map_h)

        # Update grid with visible tiles
        self.pathfinder.update_grid(ct)

        # Check arrival: adjacent to enemy core -> self-destruct
        if pos.distance_squared(Position(*self.enemy_core)) <= 2:
            ct.self_destruct()
            return

        # Get next move from pathfinder
        start = (pos.x, pos.y)
        direction = self.pathfinder.get_next_direction(start)
        if direction is None:
            return  # no path, skip turn

        # Move
        if ct.can_move(direction):
            ct.move(direction)
```

- [ ] **Step 2: Commit**

```bash
git add bots/starter/main.py
git commit -m "feat: rewrite builder bot with pathfinding toward enemy core"
```

---

### Task 4: Smoke Test via Game Match

**Files:** (none — verification only)

- [ ] **Step 1: Run a match to verify the bot loads and executes without errors**

```bash
cd /home/student/tmp && uv run cambc run bots/starter bots/starter
```

Look for: no Python tracebacks, bots spawn and attempt to move. The bot may not reach the enemy core (depends on map layout and available passable tiles), but it should not crash.

- [ ] **Step 2: If errors occur, fix and re-run**

Common issues to check:
- Import paths: `from astar import ...` vs `from .astar import ...` (depends on how cambc loads bots)
- Missing `__init__.py` if needed
- API misuse (wrong argument types, calling methods at wrong time)

- [ ] **Step 3: Commit any fixes**

```bash
git add bots/starter/
git commit -m "fix: resolve runtime issues from smoke test"
```
