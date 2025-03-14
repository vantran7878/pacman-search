from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Self, Final, Literal, Protocol
from collections.abc import Generator
from collections import deque

import heapq
import pyglet  # type: ignore
import numpy as np

type Direction = Literal['up', 'right', 'down', 'left']


class Config:
  PX_PER_UNIT: Final = 24
  COIN_RADIUS: Final = 4
  PELLET_RADIUS: Final = 8
  HEADER_HEIGHT: Final = 64
  SIDEBAR_WIDTH: Final = 512
  LINE_HEIGHT: Final = 32

  COIN_COLOR: Final = (255, 184, 151, 255)
  PATH_COLOR: Final = (0, 0, 0, 255)
  WALL_COLOR: Final = (33, 33, 222, 255)
  DOOR_COLOR: Final = COIN_COLOR
  PACMAN_COLOR: Final = (255, 255, 0, 255)

  BLINKY_COLOR: Final = (255, 0, 0, 255)
  PINKY_COLOR: Final = (255, 184, 222, 255)
  INKY_COLOR: Final = (0, 255, 222, 255)
  CLYDE_COLOR: Final = (255, 184, 71, 255)

  SIDEBAR_BG: Final = (18, 18, 18, 255)
  TPS: Final = 4


NEXT_POS: Final[dict[Direction, tuple[int, int]]] = {
  'up': (0, 1),
  'right': (1, 0),
  'down': (0, -1),
  'left': (-1, 0),
}

OPPOSITE_DIR: Final[dict[Direction, Direction]] = {
  'up': 'down',
  'right': 'left',
  'down': 'up',
  'left': 'right'
}


@dataclass(slots=True)
class FPSCounter:
  accumulated_frames: int = 0
  accumulated_time: float = 0
  fps: int = 0

  def update(self, dt: float) -> bool:
    self.accumulated_time += dt
    self.accumulated_frames += 1

    if self.accumulated_time >= 1:
      self.fps = self.accumulated_frames
      self.accumulated_time = 0
      self.accumulated_frames = 0

      return True

    return False


@dataclass(slots=True)
class Bitset2D:
  width: int
  height: int

  data: int = 0

  def index(self, x: int, y: int) -> int | None:
    if (x < 0 or x >= self.width):
      return None

    if y < 0 or y >= self.height:
      return None

    return self.width * (y + 1) - x - 1

  def add(self, x: int, y: int) -> None:
    idx = self.index(x, y)
    if idx is None:
      return
    self.data |= 1 << idx

  def get(self, x: int, y: int) -> bool:
    idx = self.index(x, y)
    if idx is None:
      return False
    return (self.data >> idx) & 1 != 0

  def remove(self, x: int, y: int) -> bool:
    idx = self.index(x, y)
    if idx is None:
      return False

    mask = (self.data >> idx) & 1
    self.data ^= mask << idx
    return mask != 0

  def display(self, chars: tuple[str, str] = ('0', '1')) -> None:
    mask = (1 << self.width) - 1
    data = self.data

    c0, c1 = chars
    table = str.maketrans('01', c0 + c1)

    for _ in range(self.height):
      row = data & mask
      print(f'{row:0{self.width}b}'.translate(table))
      data >>= self.width

  def iter(self) -> Generator[tuple[int, int]]:
    data = self.data
    while data:
      mask = data & -data
      data ^= mask

      idx = mask.bit_length() - 1
      y = idx // self.width
      x = self.width - (idx % self.width) - 1
      yield x, y

  def clear(self) -> None:
    self.data = 0

  def len(self) -> int:
    return self.data.bit_count()


@dataclass(frozen=True, slots=True)
class Map:
  walls: Bitset2D
  doors: Bitset2D

  ghost_house_start: tuple[int, int]
  ghost_house_end: tuple[int, int]


class SearchAlgorithm(Protocol):
  @classmethod
  @abstractmethod
  def new(cls, state: GameState) -> Self: ...

  @abstractmethod
  def search(
    self,
    ghost: Ghost,
    dirs: list[Direction],
    state: GameState,
  ) -> Direction: ...

  def next_dir(self, ghost: Ghost, state: GameState) -> Direction:
    ghost_x, ghost_y = ghost.pos
    directions: list[Direction] = []

    for direction, (x, y) in NEXT_POS.items():
      if direction == OPPOSITE_DIR[ghost.dir]:
        continue

      next_x, next_y = ghost_x + x, ghost_y + y

      if next_x < 0 or next_x >= state.width:
        continue

      if next_y < 0 or next_y >= state.height:
        continue

      if state.map.walls.get(next_x, next_y):
        continue

      discard = False
      for other_ghost in state.ghosts:
        if other_ghost is ghost:
          continue

        if (next_x, next_y) == other_ghost.pos:
          discard = True
          break

        other_x, other_y = other_ghost.pos
        other_dx, other_dy = NEXT_POS[other_ghost.dir]

        if (next_x, next_y) == (other_x + other_dx, other_y + other_dy):
          discard = True
          break

      if discard:
        continue

      directions.append(direction)

    if len(directions) == 0:
      return OPPOSITE_DIR[ghost.dir]

    return self.search(ghost, directions, state)


@dataclass(slots=True)
class GreedyBestFirstSearch(SearchAlgorithm):

  @classmethod
  def new(cls, state: GameState) -> Self:
    _ = state
    return cls()

  @staticmethod
  def squared_distance(
    p0: tuple[int, int],
    direction: Direction,
    p1: tuple[int, int],
  ) -> int:
    x0, y0 = p0
    x1, y1 = p1

    xd, yd = NEXT_POS[direction]
    dx = x0 + xd - x1
    dy = y0 + yd - y1

    return dx * dx + dy * dy

  def search(
    self,
    ghost: Ghost,
    dirs: list[Direction],
    state: GameState,
  ) -> Direction:
    best_direction = dirs.pop()
    best_distance = GreedyBestFirstSearch.squared_distance(
      ghost.pos,
      best_direction,
      state.pacman.pos
    )

    for direction in dirs:
      distance = GreedyBestFirstSearch.squared_distance(
        ghost.pos,
        direction,
        state.pacman.pos
      )

      if distance < best_distance:
        best_direction = direction
        best_distance = distance

    return best_direction


type GraphCostType = dict[tuple[int, int], dict[tuple[int, int], int]]


def undel_neighbor(state: GameState) -> list[tuple[int, int]]:
  dirs: list[Direction] = ['up', 'down', 'left', 'right']
  w = state.map.walls.width
  h = state.map.walls.height

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


def jps_graph(
  state: GameState
 ) -> GraphCostType:
  dirs: list[Direction] = ['up', 'down', 'left', 'right']
  cost: GraphCostType = {}
  h = state.map.walls.height
  w = state.map.walls.width

  v: list[tuple[int, int]] = undel_neighbor(state)
  for y in range(h):
    for x in range(w):
      if state.map.walls.get(x, y):
        continue

      queue: deque[tuple[int, int, int]] = deque([(x, y, 0)])
      visited = {(x, y)}

      while queue:
        cx, cy, dist = queue.popleft()
        if (cx, cy) in v and (cx, cy) != (x, y):
          if (x, y) not in cost:
            cost[(x, y)] = {}
          cost[(x, y)][(cx, cy)] = dist
          continue
        visited.add((cx, cy))

        for direction in dirs:
          dx, dy = NEXT_POS[direction]
          next_x, next_y = cx + dx, cy + dy
          if next_x < 0 or next_x >= w:
            continue
          if next_y < 0 or next_y >= h:
            continue
          if state.map.walls.get(next_x, next_y):
            continue
          if (next_x, next_y) in visited:
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
  graph_cost: GraphCostType = field(default_factory=dict)

  @classmethod
  def new(cls, state: GameState) -> Self:
    instance = cls()
    instance.graph_cost = jps_graph(state)
    return instance

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
      heapq.heappush(pq, (1, (nx, ny), direction))
      costs[(nx, ny)] = 1

      if (nx, ny) == (goal_x, goal_y):
        continue

      if (nx, ny) in distances[(goal_x, goal_y)]:
        cost_goal = 1 + distances[(goal_x, goal_y)][(nx, ny)]
        if (
            (goal_x, goal_y) not in costs or
            cost_goal < costs[(goal_x, goal_y)]
            ):
          costs[(goal_x, goal_y)] = cost_goal
          heapq.heappush(pq, (cost_goal, (goal_x, goal_y), direction))

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
            costs[(goal_x, goal_y)] = cost_goal
            heapq.heappush(pq, (cost_goal, (goal_x, goal_y), direction))

    while pq:
      cost, (x, y), direction = heapq.heappop(pq)
      if (x, y) in costs and cost != costs[(x, y)]:
        continue
      if (x, y) == state.pacman.pos:
        return direction
      for (nx, ny), (dist) in distances[(x, y)].items():
        new_cost = cost + dist
        if (nx, ny) not in costs or new_cost < costs[(nx, ny)]:
          costs[(nx, ny)] = new_cost
          heapq.heappush(pq, (new_cost, (nx, ny), direction))
          if (nx, ny) in distances[(goal_x, goal_y)]:
            cost_goal = cost + dist + distances[(goal_x, goal_y)][(nx, ny)]
            if (
                (goal_x, goal_y) not in costs or
                cost_goal < costs[(goal_x, goal_y)]
                ):
              costs[(goal_x, goal_y)] = cost_goal
              heapq.heappush(pq, (cost_goal, (goal_x, goal_y), direction))

    return best_direction


@dataclass(slots=True)
class Ghost:
  color: tuple[int, int, int, int]
  pos: tuple[int, int]
  dir: Direction
  algorithm: SearchAlgorithm
  frame: float = 0

  def update(self, dt: float, state: GameState):
    self.frame += dt * Config.TPS
    if self.frame >= 1:
      x, y = self.pos
      dx, dy = NEXT_POS[self.dir]

      self.pos = (x + dx, y + dy)
      self.dir = self.algorithm.next_dir(self, state)
      self.frame = 0


@dataclass(slots=True)
class Pacman:
  pos: tuple[int, int]
  started: bool = False
  dir: Direction | None = None
  next_dir: Direction | None = None
  frame: float = 0

  def update(self, dt: float, state: GameState):
    if not self.started:
      if self.next_dir == 'left':
        self.dir = 'left'
        x, y = self.pos
        self.pos = (x + 1, y)
        self.frame = 0.5
        self.started = True

      if self.next_dir == 'right':
        self.dir = 'right'
        self.frame = 0.5
        self.started = True

    else:
      if self.dir is not None:
        self.frame += dt * Config.TPS
        if self.frame >= 1:
          x, y = self.pos
          dx, dy = NEXT_POS[self.dir]

          x += dx
          y += dy

          match x:
            case state.width:
              self.next_dir = 'right'
              x = -1
            case -1:
              self.next_dir = 'left'
              x = state.width

            case _:
              pass

          if state.coins.remove(x, y):
            state.score += 10
            state.removed_coins.append((x, y))

          if state.pellets.remove(x, y):
            state.score += 50
            state.removed_coins.append((x, y))

          self.pos = (x, y)

          if self.next_dir is not None:
            rx, ry = NEXT_POS[self.next_dir]
            if not state.map.walls.get(x + rx, y + ry):
              self.dir = self.next_dir

          dx, dy = NEXT_POS[self.dir]
          if state.map.walls.get(x + dx, y + dy):
            self.dir = None
          self.frame = 0

      elif self.next_dir is not None:
        x, y = self.pos
        dx, dy = NEXT_POS[self.next_dir]
        if not state.map.walls.get(x + dx, y + dy):
          self.dir = self.next_dir


@dataclass(slots=True)
class GameState:
  width: int
  height: int

  map: Map
  coins: Bitset2D
  pellets: Bitset2D

  pacman: Pacman

  score: int = 0
  removed_coins: list[tuple[int, int]] = field(default_factory=list)
  ghosts: list[Ghost] = field(default_factory=list)

  def add_ghosts(self):
    self.ghosts.append(Ghost(
      color=Config.BLINKY_COLOR,
      pos=(1, self.height - 2),
      dir='right',
      algorithm=GreedyBestFirstSearch.new(self)
    ))

    self.ghosts.append(Ghost(
      color=Config.PINKY_COLOR,
      pos=(self.width - 2, self.height - 2),
      dir='left',
      algorithm=GreedyBestFirstSearch.new(self)
    ))

    self.ghosts.append(Ghost(
      color=Config.INKY_COLOR,
      pos=(1, 1),
      dir='right',
      algorithm=UniformCostSearch.new(self)
    ))

    self.ghosts.append(Ghost(
      color=Config.CLYDE_COLOR,
      pos=(self.width - 2, 1),
      dir='left',
      algorithm=GreedyBestFirstSearch.new(self)
    ))

  @classmethod
  def from_file(cls, filename: str) -> Self:
    with open(filename, encoding='utf-8') as map_file:
      lines = [line[:-1] for line in map_file]
      w = max(len(line) for line in lines)
      h = len(lines)

      doors = Bitset2D(w, h)
      walls = Bitset2D(w, h)
      coins = Bitset2D(w, h)
      pellets = Bitset2D(w, h)

      ghost_house_start = (w, h)
      ghost_house_end = (-1, -1)

      pacman_pos = None

      for lineno, line in enumerate(lines):
        for x, tile in enumerate(line):
          y = h - lineno - 1
          if tile in '#+-':
            walls.add(x, y)

          if tile == '-':
            doors.add(x, y)

          if line[x:x + 2] == 'PP':
            pacman_pos = (x, y)

          if tile == '+':
            if x <= ghost_house_start[0] and y <= ghost_house_start[1]:
              ghost_house_start = (x, y)

            if x >= ghost_house_end[0] and y >= ghost_house_end[1]:
              ghost_house_end = (x, y)

          if tile == '.':
            coins.add(x, y)

          if tile == '*':
            pellets.add(x, y)

      assert pacman_pos is not None, 'Pacman position not defined'

      return cls(
        width=w,
        height=h,
        coins=coins,
        pellets=pellets,
        pacman=Pacman(pos=pacman_pos),
        map=Map(
          walls=walls,
          doors=doors,
          ghost_house_start=ghost_house_start,
          ghost_house_end=ghost_house_end,
        ),
      )


class Renderer:
  def __init__(self, state: GameState) -> None:
    self.batch = pyglet.graphics.Batch()

    w = state.map.walls.width
    h = state.map.walls.height

    res = Config.PX_PER_UNIT

    rgba_array = np.full(
      fill_value=Config.WALL_COLOR,
      shape=(h, w, res, res, 4),
      dtype=np.uint8
    )

    for x, y in state.map.doors.iter():
      rgba_array[y, x, :, :] = Config.DOOR_COLOR

    for y in range(h):
      for x in range(w):
        road_pad = res * 1 // 8 if (
          state.map.ghost_house_start[0] < x < state.map.ghost_house_end[0] and
          state.map.ghost_house_start[1] < y < state.map.ghost_house_end[1]
        ) else res * 5 // 11

        door_pad = road_pad + res // 5

        if state.map.walls.get(x, y):
          if x - 1 >= 0 and not state.map.walls.get(x - 1, y):
            if y - 1 >= 0 and not state.map.walls.get(x, y - 1):
              li, ri = road_pad, road_pad + road_pad
              lj, rj = road_pad, road_pad + road_pad

              di = ri - np.arange(li, ri).reshape(1, -1)
              dj = rj - np.arange(lj, rj).reshape(-1, 1)

              mask = di * di + dj * dj > road_pad * road_pad
              rgba_array[y, x, li:ri, lj:rj][mask] = Config.PATH_COLOR

            if y + 1 < h and not state.map.walls.get(x, y + 1):
              li, ri = res - road_pad - road_pad, res - road_pad
              lj, rj = road_pad, road_pad + road_pad

              di = ri - np.arange(li, ri).reshape(1, -1)
              dj = np.arange(lj, rj).reshape(-1, 1) - lj + 1

              mask = di * di + dj * dj > road_pad * road_pad
              rgba_array[y, x, li:ri, lj:rj][mask] = Config.PATH_COLOR

          if x + 1 < w and not state.map.walls.get(x + 1, y):
            if y - 1 >= 0 and not state.map.walls.get(x, y - 1):
              li, ri = road_pad, road_pad + road_pad
              lj, rj = res - road_pad - road_pad, res - road_pad

              di = np.arange(li, ri).reshape(1, -1) - li + 1
              dj = rj - np.arange(lj, rj).reshape(-1, 1)

              mask = di * di + dj * dj > road_pad * road_pad
              rgba_array[y, x, li:ri, lj:rj][mask] = Config.PATH_COLOR

            if y + 1 < h and not state.map.walls.get(x, y + 1):
              li, ri = res - road_pad - road_pad, res - road_pad
              lj, rj = res - road_pad - road_pad, res - road_pad

              dj = np.arange(li, ri).reshape(1, -1) - li + 1
              di = np.arange(lj, rj).reshape(-1, 1) - lj + 1

              mask = di * di + dj * dj > road_pad * road_pad
              rgba_array[y, x, li:ri, lj:rj][mask] = Config.PATH_COLOR
        else:
          rgba_array[y, x] = Config.PATH_COLOR

          if x - 1 >= 0:
            pad = door_pad if state.map.doors.get(x - 1, y) else road_pad
            rgba_array[y, x - 1, :, res - pad:] = Config.PATH_COLOR
          if x + 1 < w:
            pad = door_pad if state.map.doors.get(x + 1, y) else road_pad
            rgba_array[y, x + 1, :, :pad] = Config.PATH_COLOR
          if y - 1 >= 0:
            pad = door_pad if state.map.doors.get(x, y - 1) else road_pad
            rgba_array[y - 1, x, res - pad:, :] = Config.PATH_COLOR
          if y + 1 < h:
            pad = door_pad if state.map.doors.get(x, y + 1) else road_pad
            rgba_array[y + 1, x, :pad, :] = Config.PATH_COLOR

          if x - 1 >= 0:
            if y - 1 >= 0:
              li, ri = res - road_pad, res
              lj, rj = res - road_pad, res

              di = ri - np.arange(li, ri).reshape(1, -1)
              dj = rj - np.arange(lj, rj).reshape(-1, 1)

              mask = di * di + dj * dj <= road_pad * road_pad
              rgba_array[y - 1, x - 1, li:ri, lj:rj][mask] = Config.PATH_COLOR

            if y + 1 < h:
              li, ri = 0, road_pad
              lj, rj = res - road_pad, res

              di = ri - np.arange(li, ri).reshape(1, -1)
              dj = np.arange(lj, rj).reshape(-1, 1) - lj + 1

              mask = di * di + dj * dj <= road_pad * road_pad
              rgba_array[y + 1, x - 1, li:ri, lj:rj][mask] = Config.PATH_COLOR

          if x + 1 < w:
            if y - 1 >= 0:
              li, ri = res - road_pad, res
              lj, rj = 0, road_pad

              di = np.arange(li, ri).reshape(1, -1) - li + 1
              dj = rj - np.arange(lj, rj).reshape(-1, 1)

              mask = di * di + dj * dj <= road_pad * road_pad
              rgba_array[y - 1, x + 1, li:ri, lj:rj][mask] = Config.PATH_COLOR

            if y + 1 < h:
              li, ri = 0, road_pad
              lj, rj = 0, road_pad

              di = np.arange(li, ri).reshape(1, -1) - li + 1
              dj = np.arange(lj, rj).reshape(-1, 1) - lj + 1

              mask = di * di + dj * dj <= road_pad * road_pad
              rgba_array[y + 1, x + 1, li:ri, lj:rj][mask] = Config.PATH_COLOR

    rgba_bytes = rgba_array.transpose((0, 2, 1, 3, 4)).tobytes()

    image_data = pyglet.image.ImageData(
      width=w * res,
      height=h * res,
      fmt='RGBA',
      data=rgba_bytes
    )
    self.texture = image_data.get_texture()
    self.sprite = pyglet.sprite.Sprite(
      self.texture,
      batch=self.batch,
    )

    self.coins = {
      (x, y): pyglet.shapes.Circle(
        x=(x * 2 + 1) * Config.PX_PER_UNIT // 2,
        y=(y * 2 + 1) * Config.PX_PER_UNIT // 2,
        color=Config.COIN_COLOR,
        radius=Config.COIN_RADIUS,
        batch=self.batch,
      )
      for x, y in state.coins.iter()
    }

    self.pellets = {
      (x, y): pyglet.shapes.Circle(
        x=(x * 2 + 1) * Config.PX_PER_UNIT // 2,
        y=(y * 2 + 1) * Config.PX_PER_UNIT // 2,
        color=Config.COIN_COLOR,
        radius=Config.PELLET_RADIUS,
        batch=self.batch,
      )
      for x, y in state.pellets.iter()
    }

    self.pacman = pyglet.shapes.Circle(
      x=0,
      y=0,
      radius=Config.PX_PER_UNIT * 4 // 5,
      color=Config.PACMAN_COLOR,
      batch=self.batch,
    )

    self.ghosts = [
      pyglet.shapes.Circle(
        x=0,
        y=0,
        radius=Config.PX_PER_UNIT * 4 // 5,
        color=ghost.color,
        batch=self.batch
      )
      for ghost in state.ghosts
    ]

    self.sidebar_background = pyglet.shapes.Rectangle(
      x=w * Config.PX_PER_UNIT,
      y=0,
      width=Config.SIDEBAR_WIDTH,
      height=h * Config.PX_PER_UNIT + Config.HEADER_HEIGHT,
      color=Config.SIDEBAR_BG,
      batch=self.batch,
    )

    self.score_label = pyglet.text.Label(
      'Score: 0',
      font_size=Config.HEADER_HEIGHT // 3,
      x=w * Config.PX_PER_UNIT // 2,
      y=h * Config.PX_PER_UNIT + Config.HEADER_HEIGHT // 2,
      color=Config.PACMAN_COLOR,
      anchor_x='center',
      anchor_y='center',
      batch=self.batch,
    )

    self.stats_title = pyglet.text.Label(
      'Statistics',
      font_size=Config.HEADER_HEIGHT // 3,
      x=w * Config.PX_PER_UNIT + Config.SIDEBAR_WIDTH // 2,
      y=h * Config.PX_PER_UNIT + Config.HEADER_HEIGHT // 2,
      color=Config.PACMAN_COLOR,
      anchor_y='center',
      anchor_x='center',
      batch=self.batch,
    )

    self.fps_counter = pyglet.text.Label(
      'Frame rate: 0 FPS',
      font_size=Config.PX_PER_UNIT * 5 // 8,
      x=w * Config.PX_PER_UNIT + Config.PX_PER_UNIT,
      y=h * Config.PX_PER_UNIT - Config.LINE_HEIGHT // 2,
      anchor_y='center',
      anchor_x='left',
      batch=self.batch,
    )

  def render_coins(self, state: GameState) -> None:
    for x, y in state.removed_coins:
      self.coins.pop((x, y), None)
      self.pellets.pop((x, y), None)

    state.removed_coins = []

  def render_pacman(self, state: GameState) -> None:
    pacman_x, pacman_y = state.pacman.pos
    if state.pacman.started is False:
      self.pacman.x = (pacman_x + 1) * Config.PX_PER_UNIT
      self.pacman.y = (pacman_y * 2 + 1) * Config.PX_PER_UNIT // 2
    else:
      x, y = state.pacman.pos
      if state.pacman.dir is not None:
        dx, dy = NEXT_POS[state.pacman.dir]

        tx = (x + dx * state.pacman.frame) * 2 + 1
        ty = (y + dy * state.pacman.frame) * 2 + 1

        self.pacman.x = tx * Config.PX_PER_UNIT // 2
        self.pacman.y = ty * Config.PX_PER_UNIT // 2
      else:
        self.pacman.x = (x * 2 + 1) * Config.PX_PER_UNIT // 2
        self.pacman.y = (y * 2 + 1) * Config.PX_PER_UNIT // 2

  def render_ghosts(self, state: GameState) -> None:
    for ghost, ghost_sprite in zip(state.ghosts, self.ghosts):
      x, y = ghost.pos
      dx, dy = NEXT_POS[ghost.dir]

      tx = (x + dx * ghost.frame) * 2 + 1
      ty = (y + dy * ghost.frame) * 2 + 1

      ghost_sprite.x = tx * Config.PX_PER_UNIT // 2
      ghost_sprite.y = ty * Config.PX_PER_UNIT // 2

  def render_ui(self, state: GameState) -> None:
    self.score_label.text = f"Score: {state.score}"

  def render(self, state: GameState) -> None:
    self.render_coins(state)
    self.render_pacman(state)
    self.render_ghosts(state)
    self.render_ui(state)
    self.batch.draw()


def main():
  state = GameState.from_file('./map.txt')
  state.add_ghosts()

  window = pyglet.window.Window(
    width=Config.PX_PER_UNIT * state.width + Config.SIDEBAR_WIDTH,
    height=Config.PX_PER_UNIT * state.height + Config.HEADER_HEIGHT,
    caption='Pacman search',
  )

  renderer = Renderer(state)

  def on_key_press(symbol: int, modifiers: int):
    keymap: dict[int, Direction] = {
      pyglet.window.key.UP: 'up',
      pyglet.window.key.RIGHT: 'right',
      pyglet.window.key.DOWN: 'down',
      pyglet.window.key.LEFT: 'left',

      pyglet.window.key.W: 'up',
      pyglet.window.key.D: 'right',
      pyglet.window.key.S: 'down',
      pyglet.window.key.A: 'left',

      pyglet.window.key.K: 'up',
      pyglet.window.key.L: 'right',
      pyglet.window.key.J: 'down',
      pyglet.window.key.H: 'left',
    }

    direction = keymap.get(symbol)

    if modifiers == 0 and direction is not None:
      state.pacman.next_dir = direction

  fps_counter = FPSCounter()

  def update(dt: float):
    if fps_counter.update(dt):
      renderer.fps_counter.text = f"Frame rate: {fps_counter.fps} FPS"
    state.pacman.update(dt, state)
    for ghost in state.ghosts:
      ghost.update(dt, state)

    window.clear()
    renderer.render(state)

  def on_draw():
    pass

  pyglet.clock.schedule(update)  # type: ignore
  window.event(on_key_press)  # type: ignore
  window.event(on_draw)  # type: ignore

  pyglet.app.run()


if __name__ == '__main__':
  main()
