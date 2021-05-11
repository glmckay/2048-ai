import atexit
import curses
import numpy
import os
import time
from typing import Iterable, List, Mapping, Optional
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN

import agentutils
from game2048 import Game

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa E402


def play_game(game: Game, model: Optional["tf.keras.Model"] = None):
    def cleanup():
        curses.endwin()

    atexit.register(cleanup)

    scr_win = curses.initscr()
    curses.noecho()  # don't echo input characters
    curses.curs_set(0)  # invisible cursor

    try:
        if model:
            snake_terminal = Terminal2048WithModel(game, scr_win, model)
            snake_terminal.run_state_machine(initial_state="PAUSED")
        else:
            snake_terminal = Terminal2048(game, scr_win)
            snake_terminal.run_state_machine(initial_state="PLAY")

    finally:
        cleanup()
        atexit.unregister(cleanup)


def clamp_value(m, v, M):
    return max(m, min(v, M))


class Terminal2048:
    CELL_WIDTH = 6

    KEY_QUIT = ord("q")
    KEY_PAUSE = ord(" ")
    ACTION_KEY_MAP: Mapping[int, Game.Action] = {
        KEY_UP: "UP",
        KEY_LEFT: "LEFT",
        KEY_DOWN: "DOWN",
        KEY_RIGHT: "RIGHT",
        ord("w"): "UP",
        ord("a"): "LEFT",
        ord("s"): "DOWN",
        ord("d"): "RIGHT",
    }

    state_funcs = {
        "PLAY": "play",
        "PAUSED": "paused",
    }

    def __init__(self, game: Game, scr_win: "curses._CursesWindow"):
        self.game = game
        self.scr_win = scr_win
        self.quit = False
        self.state = ""

        self.initialize_game_window()
        self.initialize_info_window()

    def initialize_game_window(self):
        self.grid_height = 2 * self.game.height + 1
        self.grid_width = self.game.width * (self.CELL_WIDTH + 1) + 1
        # add 1 to height and 2 to width for last action indicator
        self.game_win = curses.newwin(self.grid_height + 1, self.grid_width + 2, 0, 0)
        self.game_win.keypad(True)  # interpret escape sequences (e.g. arrow keys)

        self.update_game_window(redraw_grid=True)

    def initialize_info_window(self):
        height = self.game_win.getmaxyx()[0]
        offset_x = self.game_win.getmaxyx()[1] + 2
        width = self.scr_win.getmaxyx()[1] - offset_x
        self.info_win = curses.newwin(height, width, 0, offset_x)

        pause_key_name = "space" if self.KEY_PAUSE == ord(" ") else chr(self.KEY_PAUSE)

        self.info_win.addstr(4, 0, "Controls:")
        self.info_win.addstr(5, 0, f" {chr(self.KEY_QUIT):>5} quit game")
        self.info_win.addstr(6, 0, f" {pause_key_name:>5} pause game")

    def block_value(self, value: float):
        return 2 ** int(value) if value != 0 else " " * self.CELL_WIDTH

    def cell_line(self, row: Iterable[float]):
        cells = (f"{self.block_value(v):>{self.CELL_WIDTH}}" for v in row)
        return "|" + "|".join(cells) + "|"

    def update_last_action(self):
        LAST_ACTION_INDICATORS = {
            None: (" ", " "),
            "UP": (" ", "∧"),
            "LEFT": ("<", " "),
            "DOWN": (" ", "v"),
            "RIGHT": (">", " "),
        }
        left_right, up_down = LAST_ACTION_INDICATORS[self.game.last_action]
        bot = self.grid_height
        side = self.grid_width

        for i in range(-1, 2):
            self.game_win.addch(bot, side // 2 + 2 * i, left_right)
            self.game_win.addch(bot // 2 + i, side + 1, up_down)

    def update_game_window(self, redraw_grid: bool = False):
        for i, row in enumerate(self.game.board):
            self.game_win.addstr(2 * i + 1, 0, self.cell_line(row))

        if redraw_grid:
            grid_line = "+" + "+".join(("-" * self.CELL_WIDTH,) * self.game.width) + "+"
            for i in range(self.game.height + 1):
                self.game_win.addstr(2 * i, 0, grid_line)

        self.update_last_action()

        self.game_win.refresh()

    def update_info_window(self):
        if self.game.game_over:
            self.info_win.addstr(1, 0, "GAME OVER")
        elif self.state == "PAUSED":
            self.info_win.addstr(1, 0, "PAUSED")
        else:
            self.info_win.addstr(1, 0, "")

        self.info_win.refresh()

    def do_action(self, action: Game.Action):
        self.game.do_action(action)
        self.update_game_window()
        if self.game.game_over:
            # show game over message
            self.update_info_window()

    def run_state_machine(self, initial_state: str):
        self.state = initial_state
        while self.state != "QUIT":
            if self.state not in self.state_funcs:
                raise ValueError(f"Unknown state '{self.state}'")

            self.update_info_window()
            state_func = getattr(self, self.state_funcs[self.state])
            self.state = state_func()

        # run quit function if we have one before returning
        if "QUIT" in self.state_funcs:
            getattr(self, self.state_funcs["QUIT"])(self)

    def paused(self):
        while True:
            key = self.game_win.getch()
            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE and not self.game.game_over:
                return "PLAY"

    def play(self):
        while True:
            key = self.game_win.getch()
            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE:
                return "PAUSED"
            elif key in self.ACTION_KEY_MAP:
                self.do_action(self.ACTION_KEY_MAP[key])
                if self.game.game_over:
                    return "PAUSED"


class Terminal2048WithModel(Terminal2048):

    GAME_ACTIONS: List[Game.Action] = ["UP", "LEFT", "DOWN", "RIGHT"]

    KEY_SINGLE_STEP = ord("\n")  # Enter key
    KEY_SLOWER = ord("-")
    KEY_FASTER = ord("+")
    MIN_SECONDS_PER_MOVE = 0.1
    MAX_SECONDS_PER_MOVE = 1.6

    def __init__(
        self,
        game: Game,
        scr_win: "curses._CursesWindow",
        model: "tf.keras.Model",
    ):
        super().__init__(game, scr_win)
        self.model = model
        self.next_action: Game.Action = "LEFT"  # this value should never get used
        self.last_key = None
        self.seconds_per_move = 0.2

        self.initialize_model_window()

    def initialize_model_window(self):

        self.model_win = curses.newwin(
            6, self.scr_win.getmaxyx()[1], self.game_win.getmaxyx()[0] + 1, 0
        )

        self.update_next_action()

    def initialize_info_window(self):
        super().initialize_info_window()

        step_key_str = "enter" if self.KEY_SINGLE_STEP else chr(self.KEY_SINGLE_STEP)
        self.info_win.addstr(7, 0, f" {step_key_str:>5} single step")
        self.info_win.addstr(8, 0, f" {chr(self.KEY_FASTER):>5} increase speed")
        self.info_win.addstr(9, 0, f" {chr(self.KEY_SLOWER):>5} decrease speed")

    def update_info_window(self):
        self.info_win.addstr(2, 0, f"move delay: {self.seconds_per_move}s")
        super().update_info_window()

    def update_model_window(self):

        self.model_win.erase()
        # line to separate from game area
        self.model_win.addstr(0, 0, "─" * self.model_win.getmaxyx()[1])

        board = self.game.board
        logits = self.model(numpy.expand_dims(board, axis=0))[0]  # type: ignore
        probs = tf.nn.softmax(logits)

        for i, p, l in zip(range(4), probs, logits):
            action_i = self.GAME_ACTIONS[i]
            if action_i == self.next_action:
                move_str = f">{action_i}<"
            else:
                move_str = f"{action_i} "  # ending space to match potential '<'

            line = f"{move_str:>7} [{'█' * int(20 * p):<20}] {l}"
            self.model_win.addstr(2 + i, 0, line)

        self.model_win.refresh()

    def update_next_action(self):
        self.next_action = self.GAME_ACTIONS[
            agentutils.choose_action(self.model, self.game.board)
        ]
        self.update_model_window()

    def update_play_speed(self, key: int):
        old_delay = self.seconds_per_move
        self.seconds_per_move = clamp_value(
            self.MIN_SECONDS_PER_MOVE,
            self.seconds_per_move * (0.5 if key == self.KEY_FASTER else 2),
            self.MAX_SECONDS_PER_MOVE,
        )
        self.update_info_window()
        return old_delay - self.seconds_per_move

    def get_next_key(self, timeout: Optional[float] = None):
        self.game_win.nodelay(timeout is not None)
        return self.game_win.getch()

    def do_action(self, action: Game.Action):
        super().do_action(action)
        if not self.game.game_over:
            self.update_game_window()
            self.update_next_action()

    def play(self):
        next_move_time = -1
        while True:
            now = time.thread_time()
            if next_move_time <= now:
                self.do_action(self.next_action)
                if self.game.game_over:
                    return "PAUSED"
                self.update_next_action()
                next_move_time = now + self.seconds_per_move

            key = self.get_next_key(int(next_move_time - now * 1000))

            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE:
                return "PAUSED"
            elif key in [self.KEY_FASTER, self.KEY_SLOWER]:
                next_move_time += self.update_play_speed(key)

    def paused(self):
        while True:
            key = self.get_next_key()
            if key == self.KEY_QUIT:
                return "QUIT"
            elif key == self.KEY_PAUSE and not self.game.game_over:
                return "PLAY"
            elif key in [self.KEY_FASTER, self.KEY_SLOWER]:
                self.update_play_speed(key)
            elif key in self.ACTION_KEY_MAP and not self.game.game_over:
                self.do_action(self.ACTION_KEY_MAP[key])
            elif key == self.KEY_SINGLE_STEP:
                self.do_action(self.next_action)


if __name__ == "__main__":
    play_game(Game())
