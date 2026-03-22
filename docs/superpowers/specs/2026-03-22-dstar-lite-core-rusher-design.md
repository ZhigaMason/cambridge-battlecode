# D* Lite Core-Rusher Builder Bot Design

## Overview

Replace the BUILDER_BOT's random-walk behavior with a dedicated core-rushing strategy. The bot uses pathfinding (D* Lite or A*) to navigate toward the opponent's core through already-passable tiles, then self-destructs on arrival.

## Goal Inference

The bot infers the enemy core position from its own core's position using map symmetry. Maps are horizontally symmetric, vertically symmetric, or rotationally symmetric. The bot tries all three and selects the one that doesn't land on a friendly tile:

- Horizontal: `(x, height - 1 - y)`
- Vertical: `(width - 1 - x, y)`
- Rotational: `(width - 1 - x, height - 1 - y)`

The core communicates its position to builder bots via markers.

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

- `update_grid(ct: Controller)` — scan visible tiles via `ct.get_nearby_tiles()`, update known passability using `ct.is_tile_passable(pos)`.
- `get_next_direction() -> Direction | None` — return the Direction to move this turn, or None if no path exists.
- `replan(start: tuple[int,int], goal: tuple[int,int])` — compute or recompute path.

## Grid / Known-World Model

Shared by both pathfinders via the base class:

- `known_tiles: dict[tuple[int,int], bool]` — maps (x, y) to passability. True = passable, False = blocked. Absent = unknown.
- Unknown tiles are treated as **impassable** (pessimistic).
- Updated each turn by iterating `ct.get_nearby_tiles()` and calling `ct.is_tile_passable(pos)`.
- Neighbors function: for a given (x, y), returns up to 8 adjacent positions within map bounds where `known_tiles.get((nx, ny)) == True`.

## A* Implementation (astar.py)

- Runs full A* from current position to goal each turn after `update_grid()`.
- 8-directional movement, uniform cost (1 per step).
- Heuristic: Chebyshev distance — `max(abs(dx), abs(dy))` — admissible and consistent for 8-directional uniform-cost movement.
- Returns the first step's Direction, or None if no path found.

## D* Lite Implementation (dstar_lite.py)

Full Koenig & Likhachev D* Lite:

- Maintains `g` and `rhs` value dicts, and a priority queue keyed by `[k1, k2]`.
- `update_grid()` detects newly discovered obstacles/passable tiles since last turn. For each changed tile, calls `update_vertex()` on affected nodes.
- `compute_shortest_path()` incrementally reprocesses only affected nodes.
- `km` accumulates as the bot moves (key modifier for start-position changes).
- Heuristic: Chebyshev distance.

## Bot Behavior Loop

Each round when `run()` is called for a BUILDER_BOT:

1. **First run:** Infer enemy core position from symmetry. Instantiate the chosen pathfinder with goal = enemy core position.
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

This covers all 8 neighbors plus the core tile itself. Checked before attempting to move each turn.

## Constraints

- Only paths through already-passable tiles (roads, conveyors, allied cores). No road-building.
- No harvesting or marker placing — the bot is a dedicated core-rusher.
- Replaces all existing BUILDER_BOT behavior.
