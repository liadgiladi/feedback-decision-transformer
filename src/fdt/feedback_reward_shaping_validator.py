import cv2
import numpy as np


class FeedbackRewardShapingValidator:
    def __init__(self, augment_only_sparse_reward_with_synthetic_oracle_feedback, verbose: int = 0):
        self.game_started = False
        self.augment_only_sparse_reward_with_synthetic_oracle_feedback = augment_only_sparse_reward_with_synthetic_oracle_feedback
        self.state = None
        self.action = None
        self.reward = None
        self.terminal = None
        self.trajectory_num = 1
        self.verbose = verbose
        self.align_first_trajectory = False  # in breakout for example, the first trajectory is not started from the begining of the game, need to wait to finish and then mark as aligned

    def is_feedback_reward_shaping_allow(self):
        if not self.game_started:
            return False

        if self.augment_only_sparse_reward_with_synthetic_oracle_feedback and self.reward != 0:
            return False

        if self.terminal:
            return False

        return True

    def handle_step(self, state, action, reward, terminal, next_state, next_action, index):
        if state[0].sum() + state[1].sum() + state[2].sum() == 0 and state[3].sum() != 0:  # indicates start of game, first state
            self.game_started = False
            self.align_first_trajectory = True

        if not self.align_first_trajectory:
            return

        if state[3].sum() == 0:
            self.game_started = False
            return

        if self.verbose:
            if state[3].sum() > 0:
                cv2.imwrite(
                    f"orig_trajectory_{self.trajectory_num}_transition_{index}_action_{action}_terminal_{terminal}.png",
                    state[3])
            elif state[2].sum() > 0:
                cv2.imwrite(
                    f"orig_trajectory_{self.trajectory_num}_transition_{index}_action_{action}_terminal_{terminal}.png",
                    state[2])
            elif state[1].sum() > 0:
                cv2.imwrite(
                    f"orig_trajectory_{self.trajectory_num}_transition_{index}_action_{action}_terminal_{terminal}.png",
                    state[1])
            elif state[0].sum() > 0:
                cv2.imwrite(
                    f"orig_trajectory_{self.trajectory_num}_transition_{index}_action_{action}_terminal_{terminal}.png",
                    state[0])

        # override current state/action/reward over previous
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal

        if terminal:
            self.trajectory_num += 1


class SeaquestFeedbackRewardShapingValidator(FeedbackRewardShapingValidator):
    def __init__(self, augment_only_sparse_reward_with_synthetic_oracle_feedback: bool):
        super(SeaquestFeedbackRewardShapingValidator, self).__init__(augment_only_sparse_reward_with_synthetic_oracle_feedback)

        self.game_started = False
        self.lives = 3
        self.is_life_loss = False
        self.steps_to_wait_after_oxygen_is_done = 45
        self.oxygen_done_wait_counter = 0

    def is_feedback_reward_shaping_allow(self, enable_sticky_action_logic: bool = True):
        if not super(SeaquestFeedbackRewardShapingValidator, self).is_feedback_reward_shaping_allow():
            return False

        if not self.align_first_trajectory:
            return False

        if self.oxygen_done_wait_counter > 0:
            return False

        if self._is_oxygen_empty(self.state):
            return False

        return True

    def _is_oxygen_empty(self, state):
        return np.all(state[3][68:70, 26:58] == 85)

    def _is_oxygen_full(self, state):
        return np.all(state[3][68:70, 26:58] == 214) and np.all(state[3][68:70, 58] == 200)

    def _count_lives_score(self):
        count = 0
        if self.state[3][10, 32] > 64:
            count += 1

        if self.state[3][10, 36] > 64:
            count += 1

        if self.state[3][10, 40] > 64:
            count += 1

        return count

    def initialize(self):
        self.oxygen_done_wait_counter = 0
        self.lives = 3
        self.game_started = False
        self.is_life_loss = False

    def handle_step(self, state, action, reward, terminal, next_state, next_action, index):
        # if black screen then mark game started as false
        if state[3].sum() == 0:
            self.initialize()

            return

        if not self.game_started and self.lives == 3 and self._is_oxygen_full(state):
            self.game_started = True

            assert self.lives == 3
            assert self.oxygen_done_wait_counter == 0
            assert self.is_life_loss == False

        # override current state/action/reward over previous
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal

        self.is_life_loss = False
        if terminal:
            self.trajectory_num += 1
            self.align_first_trajectory = True

            self.initialize()
            #self.is_life_loss = False - in Seaquest first life is lost, then we get a terminal mark, thus, no need to set it to true when terminal is True
        else:
            # update counters
            if self.oxygen_done_wait_counter > 0:
                self.oxygen_done_wait_counter -= 1

            # check if it has oxygen to detect life loss
            if self.game_started and self._is_oxygen_empty(state) and not self.oxygen_done_wait_counter > 0:
                self.oxygen_done_wait_counter = self.steps_to_wait_after_oxygen_is_done
                self.is_life_loss = self.align_first_trajectory and True

            # update lives
            if self.game_started and self._count_lives_score() < self.lives:
                self.lives = self._count_lives_score()


class QbertFeedbackRewardShapingValidator(FeedbackRewardShapingValidator):
    def __init__(self, augment_only_sparse_reward_with_synthetic_oracle_feedback: bool):
        super(QbertFeedbackRewardShapingValidator, self).__init__(augment_only_sparse_reward_with_synthetic_oracle_feedback)

        self.game_started = False
        self.lives = 3
        self.is_life_loss = False
        self.steps_to_wait_after_level_completed = 40
        self.steps_to_wait_after_death = 75
        self.steps_to_wait_after_jump_on_disc = 43
        self.cube_positions = [(14,40),
                               (25,34),(25, 49),
                               (37,28),(37,40),(37,55),
                               (48,21),(48,34),(48,49),(48,61),
                               (60,15),(60,28),(60,40),(60,55),(60,68),
                               (72,9),(72,21),(72,34),(72,49),(72,61),(72,74)]
        self.cube_colors = []
        self.prev_cube_colors = []
        self.cube_colors_second_level = []
        self.prev_cube_colors_second_level = []
        self.num_cubes = 21
        self.qbert_position_cube = None
        self.prev_qbert_position_cube = None
        self.num_steps_qbert_on_same_cube = 0
        self.qbert_was_on_cube_prev_step = -1
        self.qbert_on_cube = False

        self.level_color_before = None
        self.level_color_after = None

        self.level_completed_wait_counter = 0
        self.death_wait_counter = 0
        self.jump_disc_wait_counter = 0
        self.levels_completed = 0
        self._score_area_appeared_when_level_or_death_occurred = False

        self.next_state = None
        self.next_action = None


    def _calc_cube_colors(self, state):
        return [state[3][position] for position in self.cube_positions]

    def _calc_cube_colors_second_level(self, state):
        return [state[3][(row+1, column)] for (row, column) in self.cube_positions]

    def _is_score_area_appear(self, state):
        score_area_array = state[3][2:6, 17:30]
        score_area_array_max_value = score_area_array.max()

        return score_area_array_max_value != 0

    def _is_qbert_on_disc(self, state):
        left_disc_area = state[3][48:53, 7:12]
        if np.count_nonzero(left_disc_area == 107) >= 6 and state[3][54, 7:12].max() != 0:
            return True

        right_disc_area = state[3][48:53, 72:77]
        if np.count_nonzero(right_disc_area == 107) >= 6 and state[3][54, 72:77].max() != 0:
            return True

    def _get_score(self, state):
        if not self._is_score_area_appear(state):
            return None

        return state[3][2:6, 17:38]

    def _count_lives_score(self):
        if not self._is_score_area_appear(self.state):
            return None

        count = 0
        if self.state[3][9, 19] > 0:
            count += 1

        if self.state[3][9, 24] > 0:
            count += 1

        if self.state[3][9, 28] > 0:
            count += 1

        return count

    def _is_qbert_on_cube(self):
        if not self.level_color_after:
            return False

        if self.qbert_was_on_cube_prev_step > 0:
            return False

        if self.levels_completed > 3:
            return False

        # handle cases where qbert is on cube but the color hasn't been changed (it is the after color) due to qbert standing, example: trajectory 0: 439, 509
        if self.prev_cube_colors:
            cube_with_different_color = [i for i, (color, prev_color) in enumerate(zip(self.cube_colors, self.prev_cube_colors)) if color != prev_color]
            if len(cube_with_different_color) == 1:
                i = cube_with_different_color[0]
                if self.cube_colors[i] == self.level_color_after and self.prev_cube_colors[i] == self.level_color_before:
                    return True

        if all(color == self.level_color_before or (self.level_color_after and color == self.level_color_after) for color in self.cube_colors):
            for i, cube_position in enumerate(self.cube_positions):
                # check if qbert is currently on cube without affecting its color
                row = cube_position[0]
                column = cube_position[1]
                values = self.state[3][row-6:row, column]
                #if self.action != self.next_action and np.count_nonzero(values == 107) >= 3 and values[-1] > 90 and values[-1] < 150 and values[-2] > 50 and values[-2] < 130:
                if self.action != self.next_action and np.count_nonzero(values == 107) >= 3 and values[-1] > 90 and values[-1] < 150 and values[-2] > 70 and values[-2] < 130:
                    return True
                if self.action != self.next_action and self.next_action <= 1 and np.count_nonzero(values == 107) >= 3 and values[-1] > 90 and values[-1] < 170 and values[-2] > 50 and values[-2] < 130:
                    return True

            return False

        if self.next_state[3].sum() == 0:
            return False

        next_state_cube_colors = self._calc_cube_colors(self.next_state)
        cube_index_with_different_color = [i for i, (color, next_color) in enumerate(zip(self.cube_colors, next_state_cube_colors)) if color != next_color]
        if not len(cube_index_with_different_color):  # for cases qbert stays on the same cube multiple steps
            return True
        if len(cube_index_with_different_color) > 1:  # more than 1 disagreement implies not on cube
            return False
        if len(cube_index_with_different_color):
            i = cube_index_with_different_color[0]
            if self.action > 1 and next_state_cube_colors[i] != self.level_color_after and abs(int(next_state_cube_colors[i]) - int(self.cube_colors[i])) <= 10:  # movement on cube
                return True
            if self._is_score_area_appear(self.state) and self._is_score_area_appear(self.next_state):
                current_score = self._get_score(self.state)
                next_score = self._get_score(self.next_state)
                if not (current_score==next_score).all():  # handle cases when qbert land on cube, but next state is not jump and movement on cube logic didn't catch it, if score has changed, this means landing
                    return (self.prev_cube_colors_second_level and self.prev_cube_colors_second_level[i] != self.cube_colors_second_level[i])
            if i == 0 and next_state_cube_colors[i] != self.level_color_before and next_state_cube_colors[i] != self.level_color_after:
                return False
            if next_state_cube_colors[i] != self.level_color_after:  # if one color is different and on cube already, expect that next state qbert should jump
                return False

        return True

    def _get_qbert_cube_position_num(self):
        if not self.qbert_on_cube:
            return None

        position_cube = None

        for i in range(self.num_cubes):
            if self.cube_colors[i] != self.level_color_before:
                if not self.level_color_after or (self.level_color_after and self.cube_colors[i] != self.level_color_after):
                    position_cube = i
                    break

        return position_cube

    def _is_initial_game_state(self):
        return (len(set(self.cube_colors[1:])) == 1) and self.cube_colors[1] == self.level_color_before and self.cube_colors[0] != self.level_color_before

    def _is_level_completed(self, next_state):
        # gather all cube's color associated with the next state, to detect level completed since all cube colors are changing (more than 10 should)
        next_state_cube_colors = self._calc_cube_colors(next_state)

        # gather all cubes next state color that are not equal to current state
        number_of_cubes_that_have_changed_color = [next_state_cube_colors[i] for i in range(self.num_cubes) if self.cube_colors[i] != next_state_cube_colors[i]]

        # level is completed if more than 10 cubes have different color in one step diff
        return len(number_of_cubes_that_have_changed_color) > 10

    def _is_wait_counter_enabled(self):
        return self.jump_disc_wait_counter > 0 or self.level_completed_wait_counter > 0 or self.death_wait_counter > 0

    def _is_qbert_is_dead_without_count_lives_logic(self):
        if not self.game_started or self._is_wait_counter_enabled():
            return False

        return self.num_steps_qbert_on_same_cube >= 2

    def is_feedback_reward_shaping_allow(self, enable_sticky_action_logic: bool = True):
        if not super(QbertFeedbackRewardShapingValidator, self).is_feedback_reward_shaping_allow():
            return False

        if not self.align_first_trajectory:
            return False

        if self._is_wait_counter_enabled():
            return False

        if self.levels_completed > 3:
            return False

        if not self.qbert_on_cube:
            return False

        return True

    def initialize(self):
        self.death_wait_counter = 0
        self.level_completed_wait_counter = 0
        self.jump_disc_wait_counter = 0
        self.level_color_before = None
        self.level_color_after = None
        self.prev_cube_colors = None
        self.cube_colors = None
        self.cube_colors_second_level = None
        self.prev_cube_colors_second_level = None
        self.lives = 3
        self.is_life_loss = False
        self.qbert_position_cube = None
        self.prev_qbert_position_cube = None
        self.num_steps_qbert_on_same_cube = 0
        self.game_started = False
        self._score_area_appeared_when_level_or_death_occurred = False
        self.qbert_was_on_cube_prev_step = -1
        self.levels_completed = 0
        self.qbert_on_cube = False

    def handle_step(self, state, action, reward, terminal, next_state, next_action, index):
        # validation
        assert (self.level_completed_wait_counter > 0 and self.death_wait_counter > 0 and self.jump_disc_wait_counter > 0) == False

        # if black screen then mark game started as false
        if state[3].sum() == 0:
            self.initialize()

            return

        # calc cube colors and save prev
        self.prev_cube_colors = self.cube_colors
        self.cube_colors = self._calc_cube_colors(state)
        self.prev_cube_colors_second_level = self.cube_colors_second_level
        self.cube_colors_second_level = self._calc_cube_colors_second_level(state)

        # if counter is enabled due to death or level completion,
        # try to notify about the first movement of qbert to reset counters and follow game logic and
        # not start the heuristic logic after qbert already started his movement
        if self.prev_cube_colors and self._is_wait_counter_enabled() and self._score_area_appeared_when_level_or_death_occurred and not self._is_score_area_appear(state):
            self.level_completed_wait_counter = 0
            self.death_wait_counter = 0
            self.jump_disc_wait_counter = 0
            self._score_area_appeared_when_level_or_death_occurred = False

        if self._is_wait_counter_enabled() and len([i for i, (color, prev_color) in enumerate(zip(self.cube_colors, self.prev_cube_colors)) if color != prev_color]) == 1:
            self.death_wait_counter = 0
            self.level_completed_wait_counter = 0
            self._score_area_appeared_when_level_or_death_occurred = False

        if not self.level_color_before and not self._is_wait_counter_enabled():
            self.level_color_before = max(set(self.cube_colors), key=self.cube_colors.count)

        if self.level_color_before and not self.level_color_after and not self._is_wait_counter_enabled():
            level_color_after_list = [color for color in self.cube_colors if color != self.level_color_before]
            if len(level_color_after_list):
                most_frequent_color = max(level_color_after_list, key=level_color_after_list.count)
                if level_color_after_list.count(most_frequent_color) >= 2:
                    self.level_color_after = most_frequent_color

        if not self.game_started and self._is_initial_game_state():
            self.game_started = True
            self.level_color_after = None

            assert self.lives == 3
            assert self.death_wait_counter == 0
            assert self.level_completed_wait_counter == 0
            assert self.jump_disc_wait_counter == 0
            assert self.level_color_before is not None
            assert self.level_color_after is None
            assert self.num_steps_qbert_on_same_cube == 0
            assert self.qbert_on_cube == False
            assert self.is_life_loss == False

        # override current state/action/reward over previous
        self.state = state
        self.next_state = next_state
        self.action = action
        self.next_action = next_action
        self.reward = reward
        self.terminal = terminal

        if not self.game_started:
            return

        self.is_life_loss = False
        if terminal:
            self.trajectory_num += 1
            self.align_first_trajectory = True

            self.initialize()
            self.is_life_loss = True
        else:
            if not self._is_wait_counter_enabled() and self._is_qbert_on_cube():
                # mark qbert on cube
                self.qbert_on_cube = True

                # track qbert cube position
                self.prev_qbert_position_cube = self.qbert_position_cube
                self.qbert_position_cube = self._get_qbert_cube_position_num()

                # count num steps qbert is  on the same cube or reset counter otherwise
                if self.prev_qbert_position_cube is not None and self.qbert_position_cube is not None:
                    if self.action > 1 and self.qbert_position_cube == self.prev_qbert_position_cube and not self._is_wait_counter_enabled():
                        self.num_steps_qbert_on_same_cube += 1
                    elif self.qbert_position_cube != self.prev_qbert_position_cube:
                        self.num_steps_qbert_on_same_cube = 0

                self.qbert_was_on_cube_prev_step = 3  # assume the first mark of qbert on cube is the main step and not any movement on the cube afterwards
            else:
                self.qbert_was_on_cube_prev_step -= 1
                self.qbert_on_cube = False


            # update counters
            if self._is_wait_counter_enabled():
                if self.level_completed_wait_counter > 0:
                    self.level_completed_wait_counter -= 1
                if self.death_wait_counter > 0:
                    self.death_wait_counter -= 1
                if self.jump_disc_wait_counter > 0:
                    self.jump_disc_wait_counter -= 1
            else:
                self._score_area_appeared_when_level_or_death_occurred = False

            # check if qbert on disc
            if self._is_qbert_on_disc(state) and not self.jump_disc_wait_counter > 0:
                self.jump_disc_wait_counter = self.steps_to_wait_after_jump_on_disc
                self.qbert_position_cube = None
                self.prev_qbert_position_cube = None
                self.num_steps_qbert_on_same_cube = 0

            # check for life loss based on score area
            if self._is_score_area_appear(self.state) and self._count_lives_score() < self.lives and not self.death_wait_counter > 0:
                self.lives = self._count_lives_score()
                self.is_life_loss = True
                self.death_wait_counter = self.steps_to_wait_after_death
                self._score_area_appeared_when_level_or_death_occurred = self._is_score_area_appear(self.state)
                self.qbert_position_cube = None
                self.prev_qbert_position_cube = None
                self.num_steps_qbert_on_same_cube = 0
                self.jump_disc_wait_counter = 0 # in cases of wrong identification jumping

            # check for life loss based on num_steps_qbert_on_same_cube
            if self._is_qbert_is_dead_without_count_lives_logic() and not self.death_wait_counter > 0:
                self.lives -= 1
                self.is_life_loss = True
                self.num_steps_qbert_on_same_cube = 0
                self.death_wait_counter = self.steps_to_wait_after_death
                self.qbert_position_cube = None
                self.prev_qbert_position_cube = None
                self.jump_disc_wait_counter = 0  # in cases of wrong identification jumping

            # check for level completion
            if next_state[3].sum() != 0 and self._is_level_completed(next_state) and not self.level_completed_wait_counter > 0:
                self.level_completed_wait_counter = self.steps_to_wait_after_level_completed
                self._score_area_appeared_when_level_or_death_occurred = self._is_score_area_appear(self.state)
                self.level_color_before = None
                self.level_color_after = None
                self.num_steps_qbert_on_same_cube = 0
                self.levels_completed += 1


class PongFeedbackRewardShapingValidator(FeedbackRewardShapingValidator):
    def __init__(self, augment_only_sparse_reward_with_synthetic_oracle_feedback: bool):
        super(PongFeedbackRewardShapingValidator, self).__init__(augment_only_sparse_reward_with_synthetic_oracle_feedback)

        self.ball_x_position = None
        self.prev_ball_x_position = None
        self.paddle_y_position = None
        self.next_paddle_y_position = None
        self.score_area = None

    def is_feedback_reward_shaping_allow(self, enable_sticky_action_logic: bool = True):
        if not super(PongFeedbackRewardShapingValidator, self).is_feedback_reward_shaping_allow():
            return False

        if not self.align_first_trajectory:
            return False

        # check if ball is in the game - handle negative points
        if not self._is_ball_in_screen():
            return False

        if self.ball_x_position is None or self.prev_ball_x_position is None:
            return False

        # only augment when the ball is moving right and close to the paddle
        if not self._is_ball_moving_right():
            return False

        return True

    def get_current_score_area(self):
        return self.state[3][:9, 5:30]

    def _is_ball_in_screen(self):
        return self._is_ball_in_screen_given_state(self.state)

    def _is_ball_in_screen_given_state(self, state):
        ball_area_array = state[3][14:77,11:73]
        ball_area_array_max_value = ball_area_array.max()
        return ball_area_array_max_value != 87 and ball_area_array_max_value != 107  # 107 for grey frame in the begining

    def _is_ball_moving_right(self):
        if self.ball_x_position is None or self.prev_ball_x_position is None:
            return False

        return self.ball_x_position >= self.prev_ball_x_position

    def _get_ball_y_position(self):
        if not self._is_ball_in_screen():
            return None

        ball_area_array = self.state[3][14:77,11:73]
        ball_area_array_max_value = ball_area_array.max()

        return np.where(ball_area_array == ball_area_array_max_value)[0][0] + 14  # adding 14 that reflect the top area of the score + wall to align position to the overall pixel

    def _get_ball_x_position(self):
        if not self._is_ball_in_screen():
            return None

        ball_area_array = self.state[3][14:77,11:73]
        ball_area_array_max_value = ball_area_array.max()

        return np.where(ball_area_array == ball_area_array_max_value)[1][0] + 11  # adding 11 that reflect the end of the opponent paddle to align position to the overall pixel

    def _get_paddle_position(self, state):
        paddle_area_array = state[3][14:77,74:75]

        if len(np.where(paddle_area_array == 147)[0]):
            position = np.where(paddle_area_array == 147)[0][0] + 14
        else:
            position = self._get_ball_y_position()  # fallback for special cases where ball hit the paddle on the corners, take the ball y axis position

        return position

    def _is_opponent_paddle_in_screen(self, state):
        paddle_area_array = state[3][14:77,9:10]
        paddle_area_array_max_value = paddle_area_array.max()
        return paddle_area_array_max_value != 0 and paddle_area_array_max_value != 107  # 107 for grey frame in the begining

    def _is_paddle_on_the_sides(self):
        return self.paddle_y_position >= 75 or self.paddle_y_position <= 15

    def _is_stick_action_occurred(self):
        if self.next_paddle_y_position is None or self.paddle_y_position is None:
            return False

        if self._is_paddle_on_the_sides():
            return False

        # action 0 and 1 seems useless, as nothing happens to the racket
        # action 2 & 4 makes the racket go up, and action 3 & 5 makes the racket go down
        if (self.action == 0 or self.action == 1) and (abs(self.paddle_y_position - self.next_paddle_y_position) <= 2):
            return False

        if (self.action == 2 or self.action == 4) and (self.next_paddle_y_position <= 15 or self.next_paddle_y_position <= self.paddle_y_position - 8):  # 8 is roughly the number of skip pixels without stick actions
            return False

        if (self.action == 5 or self.action == 3) and (self.next_paddle_y_position >= 75 or self.paddle_y_position + 8 <= self.next_paddle_y_position):
            return False

        return True

    def handle_step(self, state, action, reward, terminal, next_state, next_action, index):
        if state[3].sum() == 0 or not self._is_opponent_paddle_in_screen(state) or not self._is_ball_in_screen_given_state(state):
            self.game_started = False
        else:
            self.game_started = True

        # override current state/action/reward over previous
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal

        if terminal:
            self.trajectory_num += 1
            self.align_first_trajectory = True

        if terminal or not self.game_started:
            self.prev_ball_x_position = None
            self.ball_x_position = None
            self.paddle_y_position = None
            self.next_paddle_y_position = None
        else:
            self.prev_ball_x_position = self.ball_x_position
            self.ball_x_position = self._get_ball_x_position()

            self.paddle_y_position = self.next_paddle_y_position
            self.next_paddle_y_position = self._get_paddle_position(next_state)

        # check life loss
        self.is_life_loss = False
        if terminal:
            self.is_life_loss = True
            self.score_area = None
        elif self.state is not None and self._is_opponent_paddle_in_screen(self.state):
            prev_score_area = self.score_area
            self.score_area = self.get_current_score_area()
            if prev_score_area is not None and not (prev_score_area==self.score_area).all() and prev_score_area.any() and self.score_area.any():
                self.is_life_loss = self.align_first_trajectory and True
                self.score_area = None

class BreakoutFeedbackRewardShapingValidator(FeedbackRewardShapingValidator):
    def __init__(self, augment_only_sparse_reward_with_synthetic_oracle_feedback: bool):
        super(BreakoutFeedbackRewardShapingValidator, self).__init__(augment_only_sparse_reward_with_synthetic_oracle_feedback)

        self.ball_y_position = None
        self.prev_ball_y_position = None
        self.paddle_x_position = None
        self.next_paddle_x_position = None
        self.is_life_loss = False
        self.score_area = None

    def is_feedback_reward_shaping_allow(self, enable_sticky_action_logic: bool = True):
        if not super(BreakoutFeedbackRewardShapingValidator, self).is_feedback_reward_shaping_allow():
            return False

        if not self.align_first_trajectory:
            return False

        # check if ball is in the game - handle end of life cases
        if not self.is_ball_in_screen():
            return False

        if self.ball_y_position is None or self.prev_ball_y_position is None:
            return False

        # only augment when the ball is moving down and below the bricks
        if not self.is_ball_moving_down():
            return False

        return True

    def get_current_score_area(self):
        return self.state[3][:7, 50:61]

    def is_ball_in_screen(self):
        ball_area_array = self.state[3][38:75, 5:79]
        ball_area_array_max_value = ball_area_array.max()
        return ball_area_array_max_value != 0

    def is_ball_moving_down(self):
        if self.ball_y_position is None or self.prev_ball_y_position is None:
            return False

        return self.ball_y_position > self.prev_ball_y_position

    def get_ball_y_position(self):
        if not self.is_ball_in_screen():
            return None

        ball_area_array = self.state[3][38:75, 5:79]
        ball_area_array_max_value = ball_area_array.max()

        return np.where(ball_area_array == ball_area_array_max_value)[0][0] + 38

    def get_ball_x_position(self):
        if not self.is_ball_in_screen():
            return None

        ball_area_array = self.state[3][38:75, 5:79]
        ball_area_array_max_value = ball_area_array.max()

        return np.where(ball_area_array == ball_area_array_max_value)[1][0] + 5

    def get_paddle_position(self, state):
        paddle_area_array = state[3][77:78,5:79]

        if len(np.where(paddle_area_array == 22)[1]):
            position = np.where(paddle_area_array == 22)[1][0] + 5
        else:
            position = self.paddle_x_position

        return position

    def is_stick_action_occurred(self):
        if self.next_paddle_x_position is None or self.paddle_x_position is None:
            return False

        if self.is_paddle_on_the_sides():
            return False

        if (self.action == 1 or self.action == 0) and self.paddle_x_position == self.next_paddle_x_position:
            return False

        if self.action == 2 and self.paddle_x_position + 12 == self.next_paddle_x_position:
            return False

        if self.action == 3 and self.paddle_x_position - 12 == self.next_paddle_x_position:
            return False

        return True

    def is_paddle_on_the_sides(self):
        return self.paddle_x_position == 5 or self.paddle_x_position == 78

    def handle_step(self, state, action, reward, terminal, next_state, next_action, index):
        super(BreakoutFeedbackRewardShapingValidator, self).handle_step(state, action, reward, terminal, next_state, next_action, index)

        if self.state is not None and self.get_ball_y_position():
            self.game_started = True
        else:
            self.game_started = False

        if terminal or not self.game_started:
            self.prev_ball_y_position = None
            self.ball_y_position = None
            self.paddle_x_position = None
            self.next_paddle_x_position = None
        else:
            self.is_life_loss = False
            self.prev_ball_y_position = self.ball_y_position
            self.ball_y_position = self.get_ball_y_position()

            self.paddle_x_position = self.next_paddle_x_position
            self.next_paddle_x_position = self.get_paddle_position(next_state)

        # check life loss
        self.is_life_loss = False
        if terminal:
            self.is_life_loss = True
            self.score_area = None
            self.state = None
        elif self.state is not None:
            prev_score_area = self.score_area
            self.score_area = self.get_current_score_area()
            if prev_score_area is not None and not (prev_score_area==self.score_area).all() and prev_score_area.any() and self.score_area.any():
                self.is_life_loss = True
                self.score_area = None
