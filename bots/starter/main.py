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
