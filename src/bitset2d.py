from dataclasses import dataclass
from collections.abc import Generator


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
