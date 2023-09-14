from collections import deque

import atari_py
import cv2
import numpy as np
from gym.envs.atari.atari_env import ACTION_MEANING


class Env:
    def __init__(self, args):
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', args.repeat_action_probability)  # sticky actions, by default disabled
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game.lower()))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        self.no_op_max = args.no_op_max
        self.repeat_action_probability = args.repeat_action_probability
        self.rng = args.rng
        self.device = args.device
        self.screen_dims = self.ale.getScreenDims()
        self.screen_width = self.screen_dims[0]
        self.screen_height = self.screen_dims[1]
        # Stores temporary observations used for pooling over two successive frames.
        self.screen_buffer = [
            np.empty((self.screen_height, self.screen_width), dtype=np.uint8),
            np.empty((self.screen_height, self.screen_width), dtype=np.uint8)
        ]

        # initialize fire action if required
        self.fire_action = None
        for action, action_meaning in ACTION_MEANING.items():
            if action_meaning == 'FIRE' and action in self.actions:
                self.fire_action = action
                break

    def _fetch_grayscale_observation(self):
        return self.ale.getScreenGrayscale().squeeze()

    def _pool_and_resize(self, screen_buffer):
        np.maximum(screen_buffer[0], screen_buffer[1], out=screen_buffer[0])

        transformed_image = cv2.resize(screen_buffer[0],
                                       (84, 84),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return int_image

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(np.zeros((84, 84)))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()

            # trigger a fire action if required by the env
            if self.fire_action is not None:
                self.ale.act(self.fire_action)

            # Perform up to 30 random no-ops before starting
            if self.no_op_max is not None:
                for _ in range(self.rng.randint(self.no_op_max)):
                    self.ale.act(0)  # Assumes raw action 0 is always no-op
                    if self.ale.game_over():
                        self.ale.reset_game()
                        # trigger a fire action if required by the env
                        if self.fire_action is not None:
                            self.ale.act(self.fire_action)

        # Process and return "initial" state
        #observation = self._get_state()
        self.screen_buffer[0] = self._fetch_grayscale_observation()
        self.screen_buffer[1].fill(0)
        observation = self._pool_and_resize(self.screen_buffer)
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return np.stack(list(self.state_buffer), axis=0).astype(np.uint8)

    def step(self,
             action: int):
        # Repeat action 4 times, max pool over last 2 frames
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                self.screen_buffer[0] = self._fetch_grayscale_observation()
            elif t == 3:
                self.screen_buffer[1] = self._fetch_grayscale_observation()

            done = self.ale.game_over()
            if done:
                break

        # Pool the last two observations.
        observation = self._pool_and_resize(self.screen_buffer)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return np.stack(list(self.state_buffer), axis=0).astype(np.uint8), reward, done

    def check_for_life_loss_and_initialize(self):
        lives = self.ale.lives()
        if lives < self.lives and lives > 0:
            # trigger a fire action if required by the env
            if self.fire_action is not None:
                self.ale.act(self.fire_action)

        self.lives = lives

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()