from typing import Final, Self
from collections.abc import Generator
from dataclasses import dataclass
import pyglet


@dataclass
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
    return (self.data >> idx) & 1 == 0

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
      print(f"{row:0{self.width}b}".translate(table))
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


@dataclass
class Map:
  width: int
  height: int

  walls: Bitset2D
  coins: Bitset2D
  pellets: Bitset2D

  @classmethod
  def from_file(cls, filename: str) -> Self:
    with open(filename, encoding='utf-8') as map_file:
      lines = [line[:-1] for line in map_file]
      w = max(len(line) for line in lines)
      h = len(lines)

      walls = Bitset2D(w, h)
      coins = Bitset2D(w, h)
      pellets = Bitset2D(w, h)

      for y, line in enumerate(lines):
        for x, tile in enumerate(line):
          if tile in '#+-':
            walls.add(x, y)

          if tile == '.':
            coins.add(x, y)

          if tile == '*':
            pellets.add(x, y)

      return cls(
        width=w,
        height=h,
        walls=walls,
        coins=coins,
        pellets=pellets,
      )


class Config:
  PX_PER_UNIT: Final = 24
  COIN_RADIUS: Final = 4
  PELLET_RADIUS: Final = 8
  COIN_COLOR: Final = (255, 184, 151, 255)


@dataclass
class Renderer:
  batch: pyglet.graphics.Batch
  coins: list[pyglet.shapes.Circle]
  pellets: list[pyglet.shapes.Circle]

  @classmethod
  def new(cls) -> Self:
    return cls(batch=pyglet.graphics.Batch(), coins=[], pellets=[])

  def render_map(self, game_map: Map) -> None:
    coins_len = game_map.coins.len()
    pellets_len = game_map.pellets.len()

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

    coin_pad = (Config.PX_PER_UNIT - Config.COIN_RADIUS) / 2
    pellet_pad = (Config.PX_PER_UNIT - Config.PELLET_RADIUS) / 2

    for (x, y), coin in zip(game_map.coins.iter(), self.coins):
      coin.x = x * Config.PX_PER_UNIT + coin_pad
      coin.y = (game_map.height - y - 1) * Config.PX_PER_UNIT + coin_pad

    for (x, y), pellet in zip(game_map.pellets.iter(), self.pellets):
      pellet.x = x * Config.PX_PER_UNIT + pellet_pad
      pellet.y = (game_map.height - y - 1) * Config.PX_PER_UNIT + pellet_pad

  def finish(self) -> None:
    self.batch.draw()


def main():
  game_map = Map.from_file('./map.txt')
  renderer = Renderer.new()

  window_width = Config.PX_PER_UNIT * game_map.width
  window_height = Config.PX_PER_UNIT * game_map.height

  window = pyglet.window.Window(width=window_width, height=window_height)

  @window.event
  def on_draw():
    window.clear()
    renderer.render_map(game_map)
    renderer.finish()

  pyglet.app.run(0)


if __name__ == '__main__':
  main()
