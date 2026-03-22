# D* Lite Core-Rusher Builder Bot Design

## Overview

Replace the BUILDER_BOT's random-walk behavior with a dedicated core-rushing strategy. The bot uses pathfinding (D* Lite or A*) to navigate toward the opponent's core through already-passable tiles, then self-destructs on an adjacent tile.

## Goal Inference

The bot infers the enemy core position from its own core's position using map symmetry. Maps are horizontally symmetric, vertically symmetric, or rotationally symmetric.

**Core position discovery:** Builder bots spawn adjacent to the core, so on first run the bot scans nearby entities via `ct.get_nearby_entities()` and `ct.get_entity_type(id)` to find the allied core, then reads its position via `ct.get_position(id)`. Map dimensions are obtained via `ct.get_map_width()` and `ct.get_map_height()`.

**Symmetry candidates:** Given core position `(x, y)` and map dimensions `(w, h)`:

- Horizontal: `(x, h - 1 - y)`
- Vertical: `(w - 1 - x, y)`
- Rotational: `(w - 1 - x, h - 1 - y)`

**Selection:** The bot cannot validate which candidate is correct at startup (enemy core is outside vision). Instead, it picks the candidate furthest from its own core position (maximally opposite). On symmetric maps, the correct candidate will always be the furthest one. If two candidates are equidistant, rotational is preferred (most common symmetry type).

## File Structure

```
bots/starter/
  main.py        — Player class, goal inference, bot orchestration, Pathfinder base class
  astar.py       — AStarPathfinder implementation
  dstar_lite.py  — DStarLitePathfinder implementation
```

`main.py` selects the algorithm via a `USE_DSTAR_LITE = True` constant at the top.

## Pathfinder Interface (base class in main.py)

Both algorithms implement a common `Pathfinder` base class:

- `update_grid(ct: Controller)` — scan visible tiles via `ct.get_nearby_tiles()`, update known passability.
- `get_next_direction() -> Direction | None` — return the Direction to move this turn, or None if no path exists.
- `replan(start: tuple[int,int], goal: tuple[int,int])` — compute or recompute path.

## Grid / Known-World Model

Shared by both pathfinders via the base class:

- `known_tiles: dict[tuple[int,int], bool]` — maps (x, y) to passability. True = passable, False = blocked. Absent = unknown.
- Unknown tiles are treated as **impassable** (pessimistic).
- Updated each turn by iterating `ct.get_nearby_tiles()` and calling `ct.is_tile_passable(pos)` for each.
- **Occupancy note:** `is_tile_passable()` returns False for tiles occupied by other builder bots. Since we re-scan all visible tiles every turn, previously-occupied tiles will be corrected once the other bot moves away. Only out-of-vision tiles can be stale, and those are already treated as impassable.
- **Goal tile special case:** The enemy core tile itself is not passable (not allied). The pathfinder targets the 8 tiles adjacent to the enemy core rather than the core tile itself. The goal for pathfinding is any tile where `distance_squared(tile, enemy_core) <= 2`.
- Neighbors function: for a given (x, y), returns up to 8 adjacent positions within map bounds where `known_tiles.get((nx, ny)) is True`. (`None` for unknown tiles evaluates as not-True, correctly excluding them.)

## A* Implementation (astar.py)

- Runs full A* from current position to goal each turn after `update_grid()`.
- 8-directional movement, uniform cost (1 per step).
- Heuristic: Chebyshev distance — `max(abs(dx), abs(dy))` — admissible and consistent for 8-directional uniform-cost movement.
- Goal test: any node where `distance_squared(node, enemy_core) <= 2`.
- Returns the first step's Direction, or None if no path found.

## D* Lite Implementation (dstar_lite.py)

Full Koenig & Likhachev D* Lite (optimized version from the paper):

**Search direction:** D* Lite searches **backwards** from goal toward the bot's current position. The "start" node in the algorithm is the bot's current position (s_start), but the search expands from s_goal.

**Initialization:**
- `g(s) = rhs(s) = infinity` for all nodes.
- `rhs(s_goal) = 0` (goal is the seed).
- Priority queue U initialized with s_goal using `calculate_key(s_goal)`.
- `km = 0`.
- Since the goal is "any tile adjacent to enemy core," we seed all in-bounds adjacent tiles with `rhs = 0`. (Skip tiles outside map bounds.) If a seeded tile is later discovered to be impassable, the normal `update_vertex` mechanism will set its `rhs` to infinity.

**Key calculation:**
```
calculate_key(s) = [min(g(s), rhs(s)) + h(s_start, s) + km,
                    min(g(s), rhs(s))]
```
Where `h` is Chebyshev distance.

**Edge costs:** Uniform cost of 1 per step (same as A*). Infinite for impassable/unknown tiles.

**update_vertex(u):** Standard D* Lite — if u is not a goal node, update `rhs(u) = min over neighbors s' of (c(u,s') + g(s'))`. (On a symmetric uniform-cost grid, predecessors and successors are identical, so the paper's Pred(u) and Succ(u) are interchangeable.) Then update u's presence/key in the priority queue.

**compute_shortest_path():** Expand nodes from U while `U.top_key() < calculate_key(s_start)` or `rhs(s_start) != g(s_start)`.

**Extracting next step:** After `compute_shortest_path()`, the bot picks the neighbor `s'` of `s_start` that minimizes `c(s_start, s') + g(s')`.

**km accumulation:** When the bot moves from `s_old` to `s_new`: `km += h(s_old, s_new)`.

**Grid updates:** Each turn, `update_grid()` compares newly observed tile passability against `known_tiles`. For each changed tile, update edge costs for all 8 neighbors and call `update_vertex()` on affected nodes, then `compute_shortest_path()`.

## Bot Behavior Loop

Each round when `run()` is called for a BUILDER_BOT:

1. **First run:** Discover allied core via nearby entities. Infer enemy core position from symmetry. Instantiate the chosen pathfinder with goal = enemy core position.
2. **Update grid:** Call `pathfinder.update_grid(ct)` — scan visible tiles, update known-world map.
3. **Check arrival:** If `ct.get_position().distance_squared(enemy_core_pos) <= 2`, call `ct.self_destruct()` and return.
4. **Get next move:** Call `pathfinder.get_next_direction()`. If None (no known path), skip turn.
5. **Move:** If `ct.can_move(direction)`, call `ct.move(direction)`.

## Core Behavior

Unchanged from current code — spawns builder bots on adjacent tiles (up to 3).

## Self-Destruct Condition

The bot self-destructs when adjacent to the enemy core:

```python
if ct.get_position().distance_squared(enemy_core_pos) <= 2:
    ct.self_destruct()
```

This covers all 8 neighbors. Checked before attempting to move each turn.

**Damage assumption:** `self_destruct()` deals 20 damage to the tile the bot stands on. We assume this damages structures on adjacent tiles or that the engine applies area-of-effect damage. If this proves ineffective against the enemy core, the strategy would need revision (e.g., building turrets instead).

## Constraints

- Only paths through already-passable tiles (roads, conveyors, allied cores). No road-building.
- No harvesting or marker placing — the bot is a dedicated core-rusher.
- Replaces all existing BUILDER_BOT behavior.
