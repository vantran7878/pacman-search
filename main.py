from dataclasses import dataclass
from typing import Self, Final
from collections.abc import Generator

import pyglet  # type: ignore
import numpy as np


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

  def remove(self, x: int, y: int) -> None:
    idx = self.index(x, y)
    if idx is None:
      return

    mask = (self.data >> idx) & 1
    self.data ^= mask << idx

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

  def len(self) -> int:
    return self.data.bit_count()


@dataclass(frozen=True, slots=True)
class Map:
  walls: Bitset2D
  doors: Bitset2D

  ghost_house_start: tuple[int, int]
  ghost_house_end: tuple[int, int]


@dataclass(slots=True)
class GameState:
  width: int
  height: int

  map: Map
  coins: Bitset2D
  pellets: Bitset2D

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

      for lineno, line in enumerate(lines):
        for x, tile in enumerate(line):
          y = h - lineno - 1
          if tile in '#+-':
            walls.add(x, y)

          if tile == '-':
            doors.add(x, y)

          if tile == '+':
            if x <= ghost_house_start[0] and y <= ghost_house_start[1]:
              ghost_house_start = (x, y)

            if x >= ghost_house_end[0] and y >= ghost_house_end[1]:
              ghost_house_end = (x, y)

          if tile == '.':
            coins.add(x, y)

          if tile == '*':
            pellets.add(x, y)

      return cls(
        width=w,
        height=h,
        coins=coins,
        pellets=pellets,
        map=Map(
          walls=walls,
          doors=doors,
          ghost_house_start=ghost_house_start,
          ghost_house_end=ghost_house_end,
        )
      )


class Config:
  PX_PER_UNIT: Final = 24
  COIN_RADIUS: Final = 4
  PELLET_RADIUS: Final = 8
  COIN_COLOR: Final = (255, 184, 151, 255)
  PATH_COLOR: Final = (0, 0, 0, 255)
  WALL_COLOR: Final = (33, 33, 222, 255)
  DOOR_COLOR: Final = COIN_COLOR


class Renderer:
  def __init__(self, game_map: Map) -> None:
    self.batch = pyglet.graphics.Batch()

    self.coins: list[pyglet.shapes.Circle] = []
    self.pellets: list[pyglet.shapes.Circle] = []

    w = game_map.walls.width
    h = game_map.walls.height

    res = Config.PX_PER_UNIT

    rgba_array = np.full(
      fill_value=Config.WALL_COLOR,
      shape=(h, w, res, res, 4),
      dtype=np.uint8
    )

    for x, y in game_map.doors.iter():
      rgba_array[y, x, :, :] = Config.DOOR_COLOR

    for y in range(h):
      for x in range(w):
        road_pad = res * 1 // 8 if (
          game_map.ghost_house_start[0] < x < game_map.ghost_house_end[0] and
          game_map.ghost_house_start[1] < y < game_map.ghost_house_end[1]
        ) else res * 5 // 11

        door_pad = road_pad + res // 5

        if game_map.walls.get(x, y):
          if x - 1 >= 0 and not game_map.walls.get(x - 1, y):
            if y - 1 >= 0 and not game_map.walls.get(x, y - 1):
              for i in range(road_pad, road_pad + road_pad):
                for j in range(road_pad, road_pad + road_pad):
                  di = road_pad + road_pad - i
                  dj = road_pad + road_pad - j

                  if di * di + dj * dj > road_pad * road_pad:
                    rgba_array[y, x, i, j] = Config.PATH_COLOR
            if y + 1 < h and not game_map.walls.get(x, y + 1):
              for i in range(res - road_pad - road_pad, res - road_pad):
                for j in range(road_pad, road_pad + road_pad):
                  di = i - (res - road_pad - road_pad) + 1
                  dj = road_pad + road_pad - j
                  if di * di + dj * dj > road_pad * road_pad:
                    rgba_array[y, x, i, j] = Config.PATH_COLOR

          if x + 1 < w and not game_map.walls.get(x + 1, y):
            if y - 1 >= 0 and not game_map.walls.get(x, y - 1):
              for i in range(road_pad, road_pad + road_pad):
                for j in range(res - road_pad - road_pad, res - road_pad):
                  di = road_pad + road_pad - i
                  dj = j - (res - road_pad - road_pad) + 1
                  if di * di + dj * dj > road_pad * road_pad:
                    rgba_array[y, x, i, j] = Config.PATH_COLOR

            if y + 1 < h and not game_map.walls.get(x, y + 1):
              for i in range(res - road_pad - road_pad, res - road_pad):
                for j in range(res - road_pad - road_pad, res - road_pad):
                  di = i - (res - road_pad - road_pad) + 1
                  dj = j - (res - road_pad - road_pad) + 1
                  if di * di + dj * dj > road_pad * road_pad:
                    rgba_array[y, x, i, j] = Config.PATH_COLOR
        else:
          rgba_array[y, x, :, :] = Config.PATH_COLOR

          if x - 1 >= 0:
            pad = door_pad if game_map.doors.get(x - 1, y) else road_pad
            rgba_array[y, x - 1, :, res - pad:] = Config.PATH_COLOR
          if x + 1 < w:
            pad = door_pad if game_map.doors.get(x + 1, y) else road_pad
            rgba_array[y, x + 1, :, :pad] = Config.PATH_COLOR
          if y - 1 >= 0:
            pad = door_pad if game_map.doors.get(x, y - 1) else road_pad
            rgba_array[y - 1, x, res - pad:, :] = Config.PATH_COLOR
          if y + 1 < h:
            pad = door_pad if game_map.doors.get(x, y + 1) else road_pad
            rgba_array[y + 1, x, :pad, :] = Config.PATH_COLOR

          if x - 1 >= 0:
            if y - 1 >= 0:
              for i in range(res - road_pad, res):
                for j in range(res - road_pad, res):
                  di = res - i
                  dj = res - j

                  if di * di + dj * dj <= road_pad * road_pad:
                    rgba_array[y - 1, x - 1, i, j, :] = Config.PATH_COLOR

            if y + 1 < h:
              for i in range(road_pad):
                for j in range(res - road_pad, res):
                  di = i + 1
                  dj = res - j

                  if di * di + dj * dj <= road_pad * road_pad:
                    rgba_array[y + 1, x - 1, i, j, :] = Config.PATH_COLOR

          if x + 1 < w:
            if y - 1 >= 0:
              for i in range(res - road_pad, res):
                for j in range(road_pad):
                  di = res - i
                  dj = j + 1

                  if di * di + dj * dj <= road_pad * road_pad:
                    rgba_array[y - 1, x + 1, i, j, :] = Config.PATH_COLOR

            if y + 1 < h:
              for i in range(road_pad):
                for j in range(road_pad):
                  di = i + 1
                  dj = j + 1

                  if di * di + dj * dj <= road_pad * road_pad:
                    rgba_array[y + 1, x + 1, i, j, :] = Config.PATH_COLOR

    rgba_bytes = rgba_array.transpose((0, 2, 1, 3, 4)).tobytes()

    # Create an ImageData object
    image_data = pyglet.image.ImageData(
      width=w * res,
      height=h * res,
      fmt='RGBA',
      data=rgba_bytes
    )
    self.texture = image_data.get_texture()
    self.sprite = pyglet.sprite.Sprite(self.texture, batch=self.batch)

  def render_state(self, state: GameState) -> None:
    coins_len = state.coins.len()
    pellets_len = state.pellets.len()

    while len(self.coins) > coins_len:
      del self.coins[-1]

    while len(self.pellets) > pellets_len:
      del self.coins[-1]

    while len(self.coins) < coins_len:
      self.coins.append(pyglet.shapes.Circle(
        x=0,
        y=0,
        radius=Config.COIN_RADIUS,
        color=Config.COIN_COLOR,
        batch=self.batch
      ))

    while len(self.pellets) < pellets_len:
      self.pellets.append(pyglet.shapes.Circle(
        x=0,
        y=0,
        radius=Config.PELLET_RADIUS,
        color=Config.COIN_COLOR,
        batch=self.batch
      ))

    for (x, y), coin in zip(state.coins.iter(), self.coins):
      coin.x = (x * 2 + 1) * Config.PX_PER_UNIT // 2
      coin.y = (y * 2 + 1) * Config.PX_PER_UNIT // 2

    for (x, y), pellet in zip(state.pellets.iter(), self.pellets):
      pellet.x = (x * 2 + 1) * Config.PX_PER_UNIT // 2
      pellet.y = (y * 2 + 1) * Config.PX_PER_UNIT // 2

  def finish(self) -> None:
    self.batch.draw()


def main():
  state = GameState.from_file('./map.txt')

  window = pyglet.window.Window(
    width=Config.PX_PER_UNIT * state.width + 512,
    height=Config.PX_PER_UNIT * state.height + 64,
  )

  renderer = Renderer(state.map)

  def on_draw():
    window.clear()
    renderer.render_state(state)
    renderer.finish()

  window.event(on_draw)  # type: ignore

  pyglet.app.run()


if __name__ == '__main__':
  main()
