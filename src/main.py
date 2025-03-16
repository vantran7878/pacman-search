import pyglet  # type: ignore

from game import GameState, Ghost, Renderer, Config, Direction, NEXT_POS
from search import (
  DepthFirstSearch, BreadthFirstSearch, AStarSearch, UniformCostSearch
)


def main():
  state = GameState.from_file('./map.txt')

  state.ghosts.append(Ghost(
    color=Config.BLINKY_COLOR,
    pos=(1, state.height - 2),
    dir='right',
    algorithm=AStarSearch.new(state)
  ))

  state.ghosts.append(Ghost(
    color=Config.PINKY_COLOR,
    pos=(state.width - 2, state.height - 2),
    dir='left',
    algorithm=BreadthFirstSearch.new(state)
  ))

  state.ghosts.append(Ghost(
    color=Config.INKY_COLOR,
    pos=(1, 1),
    dir='right',
    algorithm=DepthFirstSearch.new(state)
  ))

  state.ghosts.append(Ghost(
    color=Config.CLYDE_COLOR,
    pos=(state.width - 2, 1),
    dir='left',
    algorithm=UniformCostSearch.new(state)
  ))

  window_width = Config.PX_PER_UNIT * state.width
  window_height = Config.PX_PER_UNIT * state.height + Config.HEADER_HEIGHT

  window = pyglet.window.Window(
    width=window_width,
    height=window_height,
    caption='Pacman search',
  )

  renderer = Renderer(state)

  game_over_batch = pyglet.graphics.Batch()

  game_over_bg = pyglet.shapes.Rectangle(
    x=0,
    y=0,
    width=window_width,
    height=window_height,
    batch=game_over_batch,
    color=(0, 0, 0, 192)
  )

  game_over_label = pyglet.text.Label(
    'Game over',
    font_size=2 * Config.PX_PER_UNIT,
    x=window_width // 2,
    y=window_height // 2,
    color=Config.PACMAN_COLOR,
    anchor_x='center',
    anchor_y='center',
    batch=game_over_batch,
  )

  _ = (game_over_bg, game_over_label)

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

  def update(dt: float):
    state.pacman.update(dt, state)
    pacman_x, pacman_y = state.pacman.pos

    if state.pacman.dir is not None:
      dx, dy = NEXT_POS[state.pacman.dir]
      pacman_x += dx * state.pacman.frame
      pacman_y += dy * state.pacman.frame

    if not state.pacman.started:
      pacman_x += 0.5

    for ghost in state.ghosts:
      ghost.update(dt, state)

      ghost_x, ghost_y = ghost.pos
      dx, dy = NEXT_POS[ghost.dir]
      ghost_x += dx * ghost.frame
      ghost_y += dy * ghost.frame

      dx = ghost_x - pacman_x
      dy = ghost_y - pacman_y

      if dx * dx + dy * dy <= 2.56:
        state.game_over = True
        pyglet.clock.unschedule(update)  # type: ignore

  def on_draw():
    window.clear()
    renderer.render(state)

    if state.game_over:
      game_over_batch.draw()

  pyglet.clock.schedule_interval(update, 1 / 120)  # type: ignore
  window.event(on_key_press)  # type: ignore
  window.event(on_draw)  # type: ignore

  pyglet.app.run()


if __name__ == '__main__':
  main()
