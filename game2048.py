import functools
import itertools
import numpy
import random
from typing import Literal, Optional


@functools.lru_cache(maxsize=None)
def _sliding_indices(width: int, height: int, direction: "Game.Action"):
    """Returns a list of slices that can be used to index the board for sliding"""
    if direction == "UP":
        return [numpy.s_[:, j] for j in range(width)]
    if direction == "LEFT":
        return [numpy.s_[i, :] for i in range(height)]
    if direction == "DOWN":
        return [numpy.s_[::-1, j] for j in range(width)]
    if direction == "RIGHT":
        return [numpy.s_[i, ::-1] for i in range(height)]


def _merged_blocks(cells):
    """Returns an iterator over the merged blocks of an array of cells

    The cells may be empty
    The iterator will be padded with zeroes to have the same number of elements as the
    array provided.
    """
    n = cells.size
    for value, group in itertools.groupby(filter(None, cells)):
        while next(group, None):
            n -= 1
            yield value + 1 if next(group, None) else value
    # blocks exhausted, yield 0s
    yield from itertools.repeat(0, n)


def _count_zeros(array):
    return array.size - numpy.count_nonzero(array)


class Game:
    Action = Literal["UP", "LEFT", "DOWN", "RIGHT"]

    _MOVE_HORIZONTAL = 0
    _MOVE_VERTICAL = 1

    def __init__(self, width: int = 4, height: int = 4):
        assert width >= 2 and height >= 2

        self.width = width
        self.height = height
        self.new_blocks = [1, 2]
        self.board = numpy.zeros((self.height, self.width))
        self.last_action = None  # type: Optional[Game.Action]

        # initialize board with two new blocks
        self.spawn_block()
        self.spawn_block()

    def spawn_block(self):
        """Spawn new block in random empty cell on board"""
        if _count_zeros(self.board) == 0:
            return  # no empty cells

        cell = random.choice(numpy.argwhere(self.board == 0))
        self.board[cell[0], cell[1]] = random.choice(self.new_blocks)

    def shift_board(self, direction: "Game.Action"):
        for indices in _sliding_indices(self.width, self.height, direction):
            self.board[indices] = list(_merged_blocks(self.board[indices]))

    def do_action(self, action: "Game.Action"):
        self.shift_board(action)
        self.spawn_block()

        del self.possible_moves  # clear cached value
        self.last_action = action

    @functools.cached_property
    def possible_moves(self):
        """Returns dictionary or possible moves"""
        if _count_zeros(self.board) != 0:
            # there is an empty cell
            horiz = vert = True
        else:
            # diff does a[i+1] - a[i] along an axis, so zeroes indicate matching values
            horiz = _count_zeros(numpy.diff(self.board, axis=1)) != 0
            vert = _count_zeros(numpy.diff(self.board, axis=0)) != 0
        return {"UP": vert, "LEFT": horiz, "DOWN": vert, "RIGHT": horiz}

    @property
    def game_over(self):
        return not (self.possible_moves["UP"] or self.possible_moves["LEFT"])
