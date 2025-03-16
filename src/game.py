from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Final, Literal, Self

import numpy as np
import pyglet  # type: ignore

from bitset2d import Bitset2D


class SearchAlgorithm:
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

      if not state.in_bound(next_x, next_y):
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


type Direction = Literal['up', 'right', 'down', 'left']


class Config:
  PX_PER_UNIT: Final = 24
  COIN_RADIUS: Final = 4
  PELLET_RADIUS: Final = 8
  HEADER_HEIGHT: Final = 64
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


@dataclass(frozen=True, slots=True)
class Map:
  walls: Bitset2D
  doors: Bitset2D

  ghost_house_start: tuple[int, int]
  ghost_house_end: tuple[int, int]


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

  game_over: bool = False
  score: int = 0
  removed_coins: list[tuple[int, int]] = field(default_factory=list)
  ghosts: list[Ghost] = field(default_factory=list)

  def in_bound(self, x: int, y: int) -> bool:
    if x < 0 or x >= self.width:
      return False

    if y < 0 or y >= self.height:
      return False

    return True

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

    w = state.width
    h = state.height

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
