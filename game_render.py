import math

import numpy as np
import pygame

from entity import *
from entity import GameEntity, Ghost, Hero
from agent import *
from main import PacmanGameController


class GameRender:
    def __init__(self, in_width: int, in_height: int, map: list, pacman_game: PacmanGameController):
        pygame.init()
        self.width = in_width
        self.height = in_height
        self.screen = pygame.display.set_mode((in_width, in_height))
        pygame.display.set_caption('Pacman')
        self.clock = pygame.time.Clock()
        self.done = False
        self.won = False
        self.game_objects = []
        self.walls = []
        self.cookies = []
        self.ghosts = []
        self.hero: Hero = None
        self.lives = 1
        self.score = 0
        self.map = map
        self.score_cookie_pickup = 10
        self.current_mode = GhostBehaviour.SCATTER
        self.mode_event = pygame.USEREVENT + 1
        self.pacman_game = pacman_game
        self.epoch = 0
        self.max_score = 0
        self.fps = 60

        self.cur_state = None
        self.action = None
        self.pre_score = 10
        self.training = True
        self.agent = Agent()

        self.ai_control = True
        self.counter = 0

    def tick(self, in_fps: int):
        black = (0, 0, 0)
        self.fps = in_fps
        self.handle_mode_switch()
        while not self.done:
            for game_object in self.game_objects:
                game_object.tick()
                game_object.draw()

            self.display_text(f"[Score: {self.score}] [Epoch: {self.epoch}] [Lives: {self.lives}] [AI: {self.ai_control}] [Train: {self.training}] "
                              f"[Max: {self.max_score}] [Eps: {round(self.agent.brain.epsilon, 2)}] [FPS: {self.fps}]")

            if self.hero is None: self.display_text("YOU DIED", (self.width / 2 - 200, self.height / 2 - 256), 100)
            if self.get_won(): self.display_text("YOU WON", (self.width / 2 - 256, self.height / 2 - 256), 100)
            pygame.display.flip()
            self.clock.tick(self.fps)
            self.screen.fill(black)
            self._handle_events()

        print("Game over")

    def handle_mode_switch(self):
        scatter_timing = 50
        chase_timing = 50

        if self.current_mode == GhostBehaviour.CHASE:
            self.set_current_mode(GhostBehaviour.SCATTER)
        else:
            self.set_current_mode(GhostBehaviour.CHASE)

        used_timing = scatter_timing if self.current_mode == GhostBehaviour.SCATTER else chase_timing
        pygame.time.set_timer(self.mode_event, used_timing * 1000)

    def add_game_object(self, obj: GameEntity):
        self.game_objects.append(obj)

    def add_cookie(self, obj: GameEntity):
        self.game_objects.append(obj)
        self.cookies.append(obj)

    def add_ghost(self, obj: GameEntity):
        self.game_objects.append(obj)
        self.ghosts.append(obj)

    def set_won(self):
        self.won = True

    def get_won(self):
        return self.won

    def add_score(self, in_score: ScoreType):
        self.score += in_score.value

    def get_hero_position(self):
        return self.hero.get_position() if self.hero != None else (0, 0)

    def set_current_mode(self, in_mode: GhostBehaviour):
        self.current_mode = in_mode

    def get_current_mode(self):
        return self.current_mode

    def end_game(self):
        if self.hero in self.game_objects:
            self.game_objects.remove(self.hero)
        self.hero = None
        self.restart()

    def restart(self, unified_size = 24, first_run=False):
        if not first_run:
            self.agent.train(batch_size=64, update_epsilon=True)
        self.epoch += 1
        if self.score > self.max_score:
            self.max_score = self.score
            self.agent.save("agent.pth")

        self.score = 0
        self.lives = 1
        self.pacman_game = PacmanGameController()
        self.game_objects.clear()
        self.walls.clear()
        for y, row in enumerate(self.pacman_game.numpy_maze):
            for x, column in enumerate(row):
                if column == 0:
                    self.add_wall(Wall(self, x, y, unified_size))

        self.cookies.clear()
        for cookie_space in self.pacman_game.cookie_spaces:
            translated = translate_to_screen(cookie_space)
            cookie = Cookie(self, translated[0] + unified_size / 2, translated[1] + unified_size / 2)
            self.add_cookie(cookie)

        self.ghosts.clear()
        for i, ghost_spawn in enumerate(self.pacman_game.ghost_spawns):
            translated = translate_to_screen(ghost_spawn)
            ghost = Ghost(self, translated[0], translated[1], unified_size, self.pacman_game,
                          self.pacman_game.ghost_colors[i % 4])
            self.add_ghost(ghost)

        pacman = Hero(self, unified_size, unified_size, unified_size)
        self.add_hero(pacman)
        self.set_current_mode(GhostBehaviour.CHASE)
        self.pre_score = 10

    def kill_pacman(self):
        if self.training:
            reward = -100
            self.pre_score = self.score
            self.agent.update(self.cur_state, self.action, reward, self.cur_state, True)

        self.lives -= 1
        self.hero.set_position(24, 24)
        self.hero.set_direction(Direction.NONE)
        if self.lives == 0:
            self.end_game()

    def display_text(self, text, in_position=(16, 0), in_size=18):
        font = pygame.font.SysFont('Arial', in_size)
        text_surface = font.render(text, False, (255, 255, 255))
        self.screen.blit(text_surface, in_position)

    def add_wall(self, obj: Wall):
        self.add_game_object(obj)
        self.walls.append(obj)

    def get_walls(self):
        return self.walls

    def get_cookies(self):
        return self.cookies

    def get_ghosts(self):
        return self.ghosts

    def get_game_objects(self):
        return self.game_objects

    def add_hero(self, in_hero):
        self.add_game_object(in_hero)
        self.hero = in_hero

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

            if event.type == self.mode_event:
                self.handle_mode_switch()

        if self.counter == 0:
            state = self.get_state()
            if state is None:
                return

            if self.cur_state is not None and self.training:
                reward = self.score - self.pre_score
                self.pre_score = self.score
                self.agent.update(self.cur_state, self.action, reward, state, False)
                self.agent.train()

            self.cur_state = state

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_q]:
            self.ai_control = not self.ai_control
        elif pressed[pygame.K_1]:
            self.agent.brain.load("agent500.pth")
            self.agent.brain.epsilon = 0.2
            print("Loading a model that earns 500 score")
        elif pressed[pygame.K_2]:
            self.agent.brain.load("agent1000.pth")
            self.agent.brain.epsilon = 0.1
            print("Loading a model that earns 1000 score")
        elif pressed[pygame.K_3]:
            self.agent.brain.load("agent2000.pth")
            self.agent.brain.epsilon = 0.05
            print("Loading a model that earns 2000 score")
        elif pressed[pygame.K_4]:
            self.agent.brain.load("agent2500.pth")
            self.agent.brain.epsilon = self.agent.brain.epsilon_min
            print("Loading a model that earns 2500 score")
        elif pressed[pygame.K_0]:
            self.agent = Agent()
            self.agent.brain.epsilon = 1
            print("Loading a new model for training")
        elif pressed[pygame.K_t]:
            self.training = not self.training
        elif pressed[pygame.K_l]:
            self.lives += 1
        elif pressed[pygame.K_k]:
            if self.lives > 1:
                self.lives -= 1
        elif pressed[pygame.K_e]:
            if (self.agent.brain.epsilon > self.agent.brain.epsilon_min):
                self.agent.brain.epsilon -= 0.001
                self.agent.brain.epsilon = max(self.agent.brain.epsilon, self.agent.brain.epsilon_min)
        elif pressed[pygame.K_r]:
            if self.agent.brain.epsilon <= 0.9:
                self.agent.brain.epsilon += 0.001
            else:
                self.agent.brain.epsilon = 1
        elif pressed[pygame.K_f]:
            if self.fps > 35:
                self.fps = self.fps - 1
            else:
                self.fps = 30
        elif pressed[pygame.K_g]:
            self.fps = self.fps + 1
        elif pressed[pygame.K_ESCAPE]:
            self.restart(24)


        if self.ai_control:
            if self.counter == 0:
                self.action = self.agent.action(state)
            if self.hero is None: return
            if self.action == 0:
                self.hero.set_direction(Direction.UP)
            elif self.action == 1:
                self.hero.set_direction(Direction.LEFT)
            elif self.action == 2:
                self.hero.set_direction(Direction.DOWN)
            elif self.action == 3:
                self.hero.set_direction(Direction.RIGHT)
        else:
            if self.hero is None: return
            if pressed[pygame.K_w] or pressed[pygame.K_UP]:
                self.hero.set_direction(Direction.UP)
                self.action = 0
            elif pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
                self.hero.set_direction(Direction.LEFT)
                self.action = 1
            elif pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
                self.hero.set_direction(Direction.DOWN)
                self.action = 2
            elif pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
                self.hero.set_direction(Direction.RIGHT)
                self.action = 3

        self.counter += 1
        if self.counter == 24:
            self.counter = 0

    def get_state(self):
        if self.hero is None:
            return None
        x, y = translate_screen_to_coord_hero(self.hero.get_position())
        ghost_positions = [translate_screen_to_coord(ghost.get_position()) for ghost in self.ghosts]
        cookie_positions = [translate_screen_to_coord(cookie.get_position()) for cookie in self.cookies]

        left_wall = 1 if self.map[y][x - 1] == 0 else 0
        right_wall = 1 if self.map[y][x + 1] == 0 else 0
        up_wall = 1 if self.map[y - 1][x] == 0 else 0
        down_wall = 1 if self.map[y + 1][x] == 0 else 0

        l_wall = 0
        for i in range(x-1, 0, -1):
            if (self.map[y][i] == 1):
                l_wall += 1
            else:
                break

        r_wall = 0
        for i in range(x+1, len(self.map[y]), 1):
            if (self.map[y][i] == 1):
                r_wall += 1
            else:
                break

        u_wall = 0
        for i in range(y - 1, 0, -1):
            if (self.map[i][x] == 1):
                u_wall += 1
            else:
                break

        d_wall = 0
        for i in range(y + 1, len(self.map), 1):
            if (self.map[i][x] == 1):
                d_wall += 1
            else:
                break
        # print(l_wall, r_wall)
        left_cook = 1 if True in [cookie[1] == y and x-l_wall-1 < cookie[0] < x for cookie in cookie_positions] and left_wall == 0 else 0
        right_cook = 1 if True in [cookie[1] == y and x+r_wall+1 > cookie[0] > x for cookie in cookie_positions] and right_wall == 0 else 0
        up_cook = 1 if True in [cookie[0] == x and y-u_wall-1 < cookie[1] < y for cookie in cookie_positions] and up_wall == 0 else 0
        down_cook = 1 if True in [cookie[0] == x and y+d_wall+1 > cookie[1] > y for cookie in cookie_positions] and down_wall == 0 else 0

        left_ghost = 1 if True in [ghost[1] == y and x-l_wall-1 < ghost[0] < x for ghost in ghost_positions] and left_wall == 0 else 0
        right_ghost = 1 if True in [ghost[1] == y and x+r_wall+1 > ghost[0] > x for ghost in ghost_positions] and right_wall == 0 else 0
        up_ghost = 1 if True in [ghost[0] == x and y-u_wall-1 < ghost[1] < y for ghost in ghost_positions] and up_wall == 0 else 0
        down_ghost = 1 if True in [ghost[0] == x and y+d_wall+1 > ghost[1] > y for ghost in ghost_positions] and down_wall == 0 else 0
        # print(left_ghost, right_ghost, up_ghost, down_ghost)
        left_turn = 1 if (1 in self.map[y - 1][x - l_wall - 1:x] or 1 in self.map[y + 1][x - l_wall - 1:x]) and left_wall == 0 else 0
        right_turn = 1 if (1 in self.map[y - 1][x:x + r_wall + 1] or 1 in self.map[y + 1][x:x + r_wall + 1]) and right_wall == 0 else 0
        up_turn = 1 if (1 in np.transpose(self.map)[x - 1][y - u_wall - 1:y] or 1 in np.transpose(self.map)[x + 1][y - u_wall - 1:y]) and up_wall == 0 else 0
        down_turn = 1 if (1 in np.transpose(self.map)[x - 1][y:y + d_wall + 1] or 1 in np.transpose(self.map)[x + 1][y:y + d_wall + 1]) and down_wall == 0 else 0

        left_dir = 1 if self.hero.current_direction == Direction.LEFT else 0
        right_dir = 1 if self.hero.current_direction == Direction.RIGHT else 0
        up_dir = 1 if self.hero.current_direction == Direction.UP else 0
        down_dir = 1 if self.hero.current_direction == Direction.DOWN else 0

        left_near_cook = 1 if self.map[y][x - 1] == 1 and (x - 1, y) in cookie_positions else 0
        right_near_cook = 1 if self.map[y][x + 1] == 1 and (x + 1, y) in cookie_positions else 0
        up_near_cook = 1 if self.map[y - 1][x] == 1 and (x, y - 1) in cookie_positions else 0
        down_near_cook = 1 if self.map[y + 1][x] == 1 and (x, y + 1) in cookie_positions else 0

        # print(left_near_cook, right_near_cook, up_near_cook, down_near_cook)
        dis = [math.sqrt((cookie[1] - y)**2 + (cookie[0] - x)**2) for cookie in cookie_positions]
        cook_x, cook_y = cookie_positions[dis.index(min(dis))] if len(dis) > 0 else (-1, -1)

        left_pos_cook = 1 if -1 < cook_x < x else 0
        right_pos_cook = 1 if cook_x > x else 0
        up_pos_cook = 1 if -1 < cook_y < y else 0
        down_pos_cook = 1 if cook_y > y else 0
        # print(left_pos_cook, right_pos_cook, up_pos_cook, down_pos_cook)
        # print(left_dir, right_dir, up_dir, down_dir)

        ghost_pos = [[1 if ghost[0] < x else 0, 1 if ghost[0] > x else 0, 1 if ghost[1] < y else 0, 1 if ghost[1] > y else 0]
                     for ghost in ghost_positions]
        ghost_pos = [element for row in ghost_pos for element in row]
        # print(ghost_pos)

        state = [
            left_wall,
            left_cook,
            left_ghost,
            left_turn,
            left_dir,
            left_near_cook,
            left_pos_cook,
            right_wall,
            right_cook,
            right_ghost,
            right_turn,
            right_dir,
            right_near_cook,
            right_pos_cook,
            up_wall,
            up_cook,
            up_ghost,
            up_turn,
            up_dir,
            up_near_cook,
            up_pos_cook,
            down_wall,
            down_cook,
            down_ghost,
            down_turn,
            down_dir,
            down_near_cook,
            down_pos_cook,
            *(ghost_pos)
        ]

        return state