from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq
import random
from typing import Self

from bitset2d import Bitset2D
from game import Direction, GameState, Ghost, NEXT_POS, SearchAlgorithm


@dataclass(slots=True)
class BreadthFirstSearch(SearchAlgorithm):

  @classmethod
  def new(cls, state: GameState) -> Self:
    _ = state
    return cls()

  def search(
    self,
    ghost: Ghost,
    dirs: list[Direction],
    state: GameState
  ) -> Direction:
    initial_x, initial_y = ghost.pos
    queue: deque[tuple[int, int, Direction]] = deque()
    visited = Bitset2D(state.width, state.height)
    visited.add(initial_x, initial_y)

    for direction in dirs:
      dx, dy = NEXT_POS[direction]
      if (initial_x + dx, initial_y + dy) == state.pacman.pos:
        return direction

      queue.append((initial_x + dx, initial_y + dy, direction))

    while queue:
      x, y, first_move = queue.popleft()

      for dx, dy in NEXT_POS.values():
        next_x, next_y = x + dx, y + dy

        if not state.in_bound(next_x, next_y):
          continue

        if state.map.walls.get(next_x, next_y):
          continue

        if visited.get(next_x, next_y):
          continue

        if (next_x, next_y) == state.pacman.pos:
          return first_move

        queue.append((next_x, next_y, first_move))
        visited.add(next_x, next_y)

    return dirs[0]


@dataclass(slots=True)
class DepthFirstSearch(SearchAlgorithm):
  stack: deque[list[Direction]]
  visited: Bitset2D

  @classmethod
  def new(cls, state: GameState) -> Self:
    return cls(stack=deque(), visited=Bitset2D(state.width, state.height))

  def search(
    self,
    ghost: Ghost,
    dirs: list[Direction],
    state: GameState
  ) -> Direction:
    x, y = ghost.pos
    self.visited.clear()
    self.visited.add(x, y)

    if len(self.stack):
      if self.stack[0][-1] in dirs:
        self.stack[0] = [x for x in self.stack[0] if x in dirs]
      else:
        self.stack.clear()

    if len(self.stack) == 0:
      random.shuffle(dirs)
      self.stack.append(dirs)

    for directions in self.stack:
      dx, dy = NEXT_POS[directions[-1]]
      x += dx
      y += dy
      self.visited.add(x, y)

    goal_x, goal_y = state.pacman.pos
    if goal_x == -1:
      goal_x = 0

    if goal_x == state.width:
      goal_x = state.width - 1

    while not self.visited.get(goal_x, goal_y):
      directions: list[Direction] = []

      it = list(NEXT_POS.items())
      random.shuffle(it)

      for d, (dx, dy) in it:
        next_x = x + dx
        next_y = y + dy

        if not state.in_bound(next_x, next_y):
          continue

        if state.map.walls.get(next_x, next_y):
          continue

        if self.visited.get(next_x, next_y):
          continue

        directions.append(d)
        if (next_x, next_y) == state.pacman.pos:
          break

      self.stack.append(directions)

      while len(self.stack[-1]) == 0:
        self.stack.pop()

        if len(self.stack) == 0:
          return dirs[0]

        dx, dy = NEXT_POS[self.stack[-1].pop()]
        # self.visited.remove(x, y)
        x -= dx
        y -= dy

      dx, dy = NEXT_POS[self.stack[-1][-1]]
      x += dx
      y += dy
      self.visited.add(x, y)

    return self.stack.popleft()[-1]


type GraphCostType = dict[tuple[int, int], dict[tuple[int, int], int]]


def undel_neighbor(state: GameState) -> list[tuple[int, int]]:
  dirs: list[Direction] = ['up', 'down', 'left', 'right']
  w = state.width
  h = state.height

  v: list[tuple[int, int]] = []
  for y in range(h):
    for x in range(w):
      if state.map.walls.get(x, y):
        continue
      count: int = 0
      for direction in dirs:
        dx, dy = NEXT_POS[direction]
        next_x: int = x + dx
        next_y: int = y + dy
        if (
            next_x < 0 or next_x >= w or
            next_y < 0 or next_y >= h or
            state.map.walls.get(next_x, next_y)
            ):
          continue
        count += 1
      if count != 2:
        v.append((x, y))
  return v


def get_jps_graph(
  state: GameState
 ) -> GraphCostType:
  dirs: list[Direction] = ['up', 'down', 'left', 'right']
  cost: GraphCostType = {}
  h = state.height
  w = state.width

  v: list[tuple[int, int]] = undel_neighbor(state)
  for y in range(h):
    for x in range(w):
      if state.map.walls.get(x, y):
        continue

      queue: deque[tuple[int, int, int]] = deque([(x, y, 0)])
      visited = Bitset2D(w, h)
      visited.add(x, y)

      while queue:
        cx, cy, dist = queue.popleft()
        if (cx, cy) in v and (cx, cy) != (x, y):
          if (x, y) not in cost:
            cost[(x, y)] = {}
          cost[(x, y)][(cx, cy)] = dist
          continue
        visited.add(cx, cy)

        for direction in dirs:
          dx, dy = NEXT_POS[direction]
          next_x, next_y = cx + dx, cy + dy
          if next_x < 0 or next_x >= w:
            continue
          if next_y < 0 or next_y >= h:
            continue
          if state.map.walls.get(next_x, next_y):
            continue
          if visited.get(next_x, next_y):
            continue
          if (next_x, next_y) in v and (next_x, next_y) != (x, y):
            if (x, y) not in cost:
              cost[(x, y)] = {}
            cost[(x, y)][(next_x, next_y)] = dist + 1
          else:
            queue.append((next_x, next_y, dist + 1))

  return cost


@dataclass(slots=True)
class UniformCostSearch(SearchAlgorithm):
  graph_cost: GraphCostType

  @classmethod
  def new(cls, state: GameState) -> Self:
    return cls(graph_cost=get_jps_graph(state))

  def search(
    self,
    ghost: Ghost,
    dirs: list[Direction],
    state: GameState,
  ) -> Direction:
    pq: list[tuple[int, tuple[int, int], Direction]] = []
    costs: dict[tuple[int, int], int] = {}
    initial_x, initial_y = ghost.pos

    goal_x, goal_y = state.pacman.pos

    if goal_x == -1:
      goal_x = 0

    if goal_x == state.width:
      goal_x = state.width - 1

    distances = self.graph_cost
    costs[(initial_x, initial_y)] = 0

    for direction in dirs:
      dx, dy = NEXT_POS[direction]
      nx, ny = initial_x + dx, initial_y + dy
      if (
        nx < 0 or nx >= state.width or
        ny < 0 or ny >= state.height or
        state.map.walls.get(nx, ny)
      ):
        continue
      heapq.heappush(pq, (1, (nx, ny), direction))
      costs[(nx, ny)] = 1

      if (nx, ny) == (goal_x, goal_x):
        continue

      if set(distances[(goal_x, goal_y)]) == set(distances[(nx, ny)]):
        for (cx, cy), dist in distances[(goal_x, goal_y)].items():
          if (cx, cy) in distances[(nx, ny)]:
            cost_n_c = distances[(nx, ny)][(cx, cy)]
            cost_g_c = distances[(goal_x, goal_y)][(cx, cy)]
            cost_goal = cost_n_c - cost_g_c
            if cost_goal < 0:
              continue
            if (
              (goal_x, goal_y) not in costs or
              cost_goal < costs[(goal_x, goal_y)]
            ):
              costs[(goal_x, goal_y)] = cost_goal + 1
              heapq.heappush(pq, (cost_goal + 1, (goal_x, goal_y), direction))

    while pq:
      cost, (x, y), direction = heapq.heappop(pq)
      if (x, y) in costs and cost != costs[(x, y)]:
        continue
      if (x, y) == state.pacman.pos:
        return direction

      if (x, y) in distances[(goal_x, goal_y)]:
        cost_goal = cost + distances[(goal_x, goal_y)][(x, y)]
        if (
          (goal_x, goal_y) not in costs or
          cost_goal < costs[(goal_x, goal_y)]
        ):
          costs[(goal_x, goal_y)] = cost_goal
          heapq.heappush(pq, (cost_goal, (goal_x, goal_y), direction))

      for (nx, ny), (dist) in distances[(x, y)].items():
        new_cost = cost + dist
        if (nx, ny) not in costs or new_cost < costs[(nx, ny)]:
          costs[(nx, ny)] = new_cost
          heapq.heappush(pq, (new_cost, (nx, ny), direction))

    return dirs[0]


@dataclass(slots=True)
class AStarSearch(SearchAlgorithm):
  graph_cost: GraphCostType

  @classmethod
  def new(cls, state: GameState) -> Self:
    return cls(graph_cost=get_jps_graph(state))

  def search(
    self,
    ghost: Ghost,
    dirs: list[Direction],
    state: GameState,
  ) -> Direction:

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
      return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority queue items: (f_score, g_score, (x, y), first_direction)
    pq: list[tuple[int, int, tuple[int, int], Direction]] = []
    costs: dict[tuple[int, int], int] = {}
    initial_x, initial_y = ghost.pos

    goal_x, goal_y = state.pacman.pos
    if goal_x == -1:
      goal_x = 0
    if goal_x == state.width:
      goal_x = state.width - 1
    goal: tuple[int, int] = (goal_x, goal_y)

    distances = self.graph_cost

    costs[(initial_x, initial_y)] = 0
    best_direction: Direction = dirs[0]

    for direction in dirs:
      dx, dy = NEXT_POS[direction]
      nx, ny = initial_x + dx, initial_y + dy
      if (
        nx < 0 or nx >= state.width or
        ny < 0 or ny >= state.height or
        state.map.walls.get(nx, ny)
      ):
        continue
      g = 1
      f = g + heuristic((nx, ny), goal)
      heapq.heappush(pq, (f, g, (nx, ny), direction))
      costs[(nx, ny)] = g

      if goal in distances and (nx, ny) in distances[goal]:
        cost_goal = 1 + distances[goal][(nx, ny)]
        if goal not in costs or cost_goal < costs[goal]:
          costs[goal] = cost_goal
          f_goal = cost_goal + heuristic(goal, goal)
          heapq.heappush(pq, (f_goal, cost_goal, goal, direction))

      if (nx, ny) == (goal_x, goal_y):
        continue

      if set(distances[(goal_x, goal_y)]) == set(distances[(nx, ny)]):
        for (cx, cy) in distances[(goal_x, goal_y)]:
          if (cx, cy) in distances[(nx, ny)]:
            cost_n_c = distances[(nx, ny)][(cx, cy)]
            cost_g_c = distances[(goal_x, goal_y)][(cx, cy)]
            cost_goal = cost_n_c - cost_g_c
            if cost_goal < 0:
              continue
            if goal not in costs or cost_goal < costs[goal]:
              costs[goal] = cost_goal + 1
              f_goal = cost_goal + heuristic(goal, goal)
              heapq.heappush(pq, (f_goal + 1, cost_goal + 1, goal, direction))

    # Main A* loop.
    while pq:
      f, g, (x, y), direction = heapq.heappop(pq)
      if g != costs.get((x, y), float('inf')):
        continue
      if (x, y) == goal:
        return direction
      if (x, y) not in distances:
        continue

      if (x, y) in distances[(goal_x, goal_y)]:
        cost_goal = g + distances[(goal_x, goal_y)][(x, y)]
        if cost_goal < 0:
          continue
        if goal not in costs or cost_goal < costs[goal]:
          costs[goal] = cost_goal
          f_goal = cost_goal + heuristic(goal, goal)
          heapq.heappush(pq, (f_goal, cost_goal, goal, direction))

      for (nx, ny), edge_cost in distances[(x, y)].items():
        new_cost = g + edge_cost

        if (nx, ny) not in costs or new_cost < costs[(nx, ny)]:
          costs[(nx, ny)] = new_cost
          new_f = new_cost + heuristic((nx, ny), goal)
          heapq.heappush(pq, (new_f, new_cost, (nx, ny), direction))
    return best_direction
