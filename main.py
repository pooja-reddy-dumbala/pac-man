from game_render import *
from entity import *
import random


class PacmanGameController:
    def __init__(self):
        self.ascii_maze = [
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "X    P       XX            X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X                          X",
            "X XXXX XX XXXXXXXX XX XXXX X",
            "X XXXX XX XXXXXXXX XX XXXX X",
            "X      XX    XX    XX      X",
            "XXXXXX XXXXX XX XXXXX XXXX X",
            "XXXXXX XXXXX XX XXXXX XXXX X",
            "XXXXXX XX     G    XX XXXX X",
            "XXXXXX XX XXXXXXXX XX XXXX X",
            "XXXXXX XX X      X XX XXXX X",
            "X  G        XXXX           X",
            "X XXXX XX X      X XX XXXXXX",
            "X XXXX XX XXXXXXXX XX XXXXXX",
            "X XXXX XX    G     XX XXXXXX",
            "X XXXX XX XXXXXXXX XX XXXXXX",
            "X XXXX XX XXXXXXXX XX XXXXXX",
            "X            XX            X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X   XX       G        XX   X",
            "XXX XX XX XXXXXXXX XX XX XXX",
            "XXX XX XX XXXXXXXX XX XX XXX",
            "X      XX    XX    XX      X",
            "X XXXXXXXXXX XX XXXXXXXXXX X",
            "X XXXXXXXXXX XX XXXXXXXXXX X",
            "X                          X",
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        ]

        self.numpy_maze = []
        self.cookie_spaces = []
        self.reachable_spaces = []
        self.ghost_spawns = []
        self.ghost_colors = [
            "images/ghost.png",
            "images/ghost_pink.png",
            "images/ghost_orange.png",
            "images/ghost_blue.png"
        ]
        self.size = (0, 0)
        self.convert_to_numpy()
        self.p = Pathfinder(self.numpy_maze)

    def request_random_path(self, in_ghost: Ghost):
        random_space = random.choice(self.reachable_spaces)
        current_maze_coord = translate_screen_to_coord(in_ghost.get_position())

        path = self.p.get_path(current_maze_coord[1], current_maze_coord[0], random_space[1],
                               random_space[0])
        test_path = [translate_to_screen(item) for item in path]
        in_ghost.set_new_path(test_path)

    def convert_to_numpy(self):
        for x, row in enumerate(self.ascii_maze):
            self.size = (len(row), x + 1)
            binary_row = []
            for y, column in enumerate(row):
                if column == "G":
                    self.ghost_spawns.append((y, x))

                if column == "X":
                    binary_row.append(0)
                else:
                    binary_row.append(1)
                    self.cookie_spaces.append((y, x))
                    self.reachable_spaces.append((y, x))

            self.numpy_maze.append(binary_row)

if __name__ == "__main__":
    unified_size = 24
    pacman_game = PacmanGameController()
    size = pacman_game.size
    game_renderer = GameRender(size[0] * unified_size, size[1] * unified_size, pacman_game.numpy_maze, pacman_game)
    game_renderer.restart(unified_size, first_run=True)
    game_renderer.tick(60)