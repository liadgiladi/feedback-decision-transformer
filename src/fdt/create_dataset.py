import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import cv2
import gin
import numpy as np

# make deterministic
from augment_dataset_with_oracle_feedback_dopamine import create_runner, FeedbackOracleModel
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
from feedback_validator import BreakoutFeedbackValidator, \
    PongFeedbackValidator, QbertFeedbackValidator, SeaquestFeedbackValidator
from fixed_replay_buffer import FixedReplayBuffer


def create_oracle_feedback_model(game):
    agent = 'c51'
    restore_ckpt = f'../synthetic_oracle/checkpoints/{agent}/{game}/1/tf_checkpoints/tf_ckpt-199'
    config = """
    atari_lib.create_atari_environment.game_name = '{}'
    WrappedReplayBuffer.replay_capacity = 300
    """.format(game)

    base_dir = os.path.join('')
    gin.parse_config(config)

    # create feedback runner
    feedback_runner = create_runner(base_dir, restore_ckpt, agent, True)

    # initialize oracle feedback model
    feedback_oracle_model = FeedbackOracleModel(feedback_runner=feedback_runner)

    return feedback_oracle_model


def store_to_file_augmented_states_with_oracle_action(game: str, state_action_feedback_cache: dict):
    # initialize oracle feedback model
    feedback_oracle_model = create_oracle_feedback_model(game)

    for (augmented_state_bytes, action_feedback_state_index) in state_action_feedback_cache.items():
        augmented_state = np.frombuffer(augmented_state_bytes, dtype=np.uint8).reshape(4, 84, 84)
        oracle_action = feedback_oracle_model.select_action(augmented_state.reshape(1, 4, 84, 84).transpose(0, 2, 3, 1))

        augmented_state = cv2.resize(augmented_state[3], (254, 254))

        cv2.imwrite(os.sep.join(["../states_with_feedback",
                                 f"{action_feedback_state_index[1]}/"]) + f"index_{action_feedback_state_index[2]}_action_{action_feedback_state_index[0]}_oracle_action_{oracle_action}_feedback_{action_feedback_state_index[1]}.png",
                    augmented_state)


def print_number_of_states_with_multiple_actions(states, actions):
    state_to_actions = defaultdict(set)

    for i, state in enumerate(states):
        action = actions[i]
        state_in_bytes = state.tobytes()
        state_to_actions[state_in_bytes].add(action)

    number_of_states_with_1_actions = 0
    number_of_states_with_2_actions = 0
    number_of_states_with_3_actions = 0
    number_of_states_with_4_actions_or_more = 0

    for state, actions in state_to_actions.items():
        if len(actions) == 1:
            number_of_states_with_1_actions += 1
        elif len(actions) == 2:
            number_of_states_with_2_actions += 1
        elif len(actions) == 3:
            number_of_states_with_3_actions += 1
        elif len(actions) >= 4:
            number_of_states_with_4_actions_or_more += 1

    print(f"number of states with 1 actions is {number_of_states_with_1_actions}")
    print(f"number of states with 2 actions is {number_of_states_with_2_actions}")
    print(f"number of states with 3 actions is {number_of_states_with_3_actions}")
    print(f"number of states with 4 or more actions is {number_of_states_with_4_actions_or_more}")
    print(f"states actions coverage is {number_of_states_with_1_actions + number_of_states_with_2_actions + number_of_states_with_3_actions + number_of_states_with_4_actions_or_more}")


def generate_distribution_from_array(arr):
    unique, counts = np.unique(arr, return_counts=True)
    return np.asarray((unique, counts)).T


def print_statistics(actions, feedbacks, state_action_positive_feedback_cache,
                     returns, stepwise_returns, rtg, done_idxs):
    with np.printoptions(precision=3, suppress=True):
        actions_dist = generate_distribution_from_array(actions)
        print("actions dist:")
        print(actions_dist)

        feedbacks_dist = generate_distribution_from_array(feedbacks)
        print("feedbacks dist:")
        print(feedbacks_dist)

        state_action_positive_feedback_cache_dist = generate_distribution_from_array(list(state_action_positive_feedback_cache.values()))
        print("positive feedback action dist:")
        print(state_action_positive_feedback_cache_dist)

        returns_dist = generate_distribution_from_array(returns)
        print("returns dist:")
        print(returns_dist)

        stepwise_returns_dist = generate_distribution_from_array(stepwise_returns)
        print("stepwise_returns dist:")
        print(stepwise_returns_dist)

        rtg_per_trajectory = [rtg[0]]
        for i in done_idxs[:-1]:
            rtg_per_trajectory.append(rtg[i])

        rtg_per_trajectory_dist = generate_distribution_from_array(rtg_per_trajectory)
        print("rtg_per_trajectory dist:")
        print(np.array2string(rtg_per_trajectory_dist, separator=', '))


@dataclass
class DatasetStatistics:
    loaded_transitions: int
    num_trajectories: int
    max_rtg: int
    max_timestep: int
    transitions_per_buffer_num: str
    num_of_augmented_transitions_with_feedback: int = field(default=0)
    num_of_augmented_transitions_with_positive_feedback: int = field(default=0)
    num_of_augmented_implicit_feedbacks: int = field(default=0)
    num_of_augmented_implicit_with_positive_feedbacks: int = field(default=0)
    state_action_positive_feedback_cache_size: int = field(default=0)


def create_feedback_validator(game: str, augment_only_sparse_reward_with_synthetic_oracle_feedback: bool):
    if game.lower() == 'breakout':
        return BreakoutFeedbackValidator(augment_only_sparse_reward_with_synthetic_oracle_feedback)
    elif game.lower() == 'pong':
        return PongFeedbackValidator(augment_only_sparse_reward_with_synthetic_oracle_feedback)
    elif game.lower() == 'qbert':
        return QbertFeedbackValidator(augment_only_sparse_reward_with_synthetic_oracle_feedback)
    elif game.lower() == 'seaquest':
        return SeaquestFeedbackValidator(augment_only_sparse_reward_with_synthetic_oracle_feedback)
    else:
        raise NotImplementedError(f"missing feedback validator for game: {game}")


@dataclass
class BufferLoaderDataset:
    buffer_obss: list = field(default_factory=lambda : [])
    buffer_actions: list = field(default_factory=lambda : [])
    buffer_return: int = field(default=0)
    buffer_feedbacks: list = field(default_factory=lambda : [])
    buffer_stepwise_auxiliary_feedbacks: list = field(default_factory=lambda : [])
    buffer_stepwise_ims_largest_smallest_actions_rtg_differences: list = field(default_factory=lambda : [])
    buffer_is_step_allowed_for_feedback: list = field(default_factory=lambda : [])
    buffer_stepwise_returns: list = field(default_factory=lambda : [])
    trajectory_step_num: int = field(default=0)
    buffer_state_action_feedback_cache: dict = field(default_factory=lambda : {})
    buffer_augmented_feedbacks_counter: int = field(default=0)
    buffer_augmented_positive_feedbacks_counter: int = field(default=0)


class DQNDatasetCreator:
    def __init__(self,
                 num_buffers: int,
                 num_steps: int,
                 game: str,
                 data_dir_prefix: str,
                 trajectories_per_buffer: int,
                 context_length: int,
                 augment_reward_with_synthetic_oracle_feedback_prob: float = 0.,
                 augment_only_sparse_reward_with_synthetic_oracle_feedback: bool = False,
                 subset_augment_reward_with_synthetic_oracle_feedback_prob: float = 0.,
                 augment_reward_with_synthetic_oracle_feedback_rng_num: int = 123,
                 subset_augment_reward_with_synthetic_oracle_feedback_rng_num: int = 0,
                 states_for_feedbacks_based_on_important_states_filename: str = "",
                 num_of_important_states: int = 10000,
                 seed: int = 123,
                 verbose: int = 0):
        print(
            f"create dataset process with num_buffers={num_buffers}, num_steps={num_steps}, game={game}, data_dir_prefix={data_dir_prefix}, trajectories_per_buffer={trajectories_per_buffer}, "
            f"context_length={context_length}, augment_reward_with_synthetic_oracle_feedback_prob={augment_reward_with_synthetic_oracle_feedback_prob}, "
            f"augment_only_sparse_reward_with_synthetic_oracle_feedback={augment_only_sparse_reward_with_synthetic_oracle_feedback}, "
            f"states_for_feedbacks_based_on_important_states_filename={states_for_feedbacks_based_on_important_states_filename},"
            f"num_of_important_states={num_of_important_states}, seed={seed} has started", flush=True)

        self.num_buffers = num_buffers
        self.num_steps = num_steps
        self.game = game
        self.data_dir_prefix = data_dir_prefix
        self.trajectories_per_buffer = trajectories_per_buffer
        self.context_length = context_length
        self.augment_reward_with_synthetic_oracle_feedback_prob = augment_reward_with_synthetic_oracle_feedback_prob
        self.augment_only_sparse_reward_with_synthetic_oracle_feedback = augment_only_sparse_reward_with_synthetic_oracle_feedback
        self.states_for_feedbacks_based_on_important_states_filename = states_for_feedbacks_based_on_important_states_filename
        self.num_of_important_states = num_of_important_states
        self.feedbacks_enabled = self.states_for_feedbacks_based_on_important_states_filename or self.augment_reward_with_synthetic_oracle_feedback_prob
        self.seed = seed
        self.verbose = verbose

        self.rng = np.random.RandomState(seed)
        self.rng_augment_reward_with_synthetic_oracle_feedback = np.random.RandomState(augment_reward_with_synthetic_oracle_feedback_rng_num)
        self.subset_rng_augment_reward_with_synthetic_oracle_feedback = np.random.RandomState(subset_augment_reward_with_synthetic_oracle_feedback_rng_num)
        self.subset_augment_reward_with_synthetic_oracle_feedback_prob = subset_augment_reward_with_synthetic_oracle_feedback_prob

        self.obss = []
        self.actions = []
        self.returns = []
        self.feedbacks = []
        self.stepwise_auxiliary_feedbacks = []
        self.is_step_allowed_for_feedback = []
        self.stepwise_ims_largest_smallest_actions_rtg_differences = []
        self.state_action_feedback_cache = {}
        self.done_idxs = []
        self.stepwise_returns = []
        self.augmented_feedbacks_counters = []
        self.augmented_positive_feedbacks_counters = []

        self.transitions_per_buffer = np.zeros(self.num_buffers, dtype=int)
        self.transition_index_per_buffer = np.zeros(self.num_buffers, dtype=int)
        self.num_trajectories = 0

        self.num_of_augmented_transitions_with_feedback = 0
        self.num_of_augmented_transitions_with_positive_feedback = 0
        self.num_of_augmented_implicit_feedbacks = 0
        self.num_of_augmented_implicit_with_positive_feedbacks = 0
        self.state_action_positive_feedback_cache = {}

        self.start_feedback_index = -1
        self.state_original_index = []

    def _create_replay_buffer(self, buffer_num, agent_dir_number: str = '1', replay_capacity: int = 100000):
        data_dir = os.sep.join([self.data_dir_prefix, self.game, agent_dir_number, 'replay_logs'])
        frb = FixedReplayBuffer(
            data_dir=data_dir,
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=replay_capacity,
            extra_storage_types=[ReplayElement('feedback', (), np.uint8)] if self.feedbacks_enabled else None)

        return frb

    def _create_reward_to_go(self):
        # -- create reward-to-go dataset
        start_index = 0
        rtg = np.zeros_like(self.stepwise_returns)
        for i in self.done_idxs:
            i = int(i)
            curr_trajectory_returns = self.stepwise_returns[start_index:i]
            for j in range(i - 1, start_index - 1, -1):  # start from i-1
                rtg_j = curr_trajectory_returns[j - start_index:i - start_index]
                rtg[j] = sum(rtg_j)
            start_index = i

        return rtg

    def _create_timestep(self):
        # -- create timestep dataset
        start_index = 0
        timesteps = np.zeros(len(self.actions) + 1, dtype=int)
        for i in self.done_idxs:
            i = int(i)
            timesteps[start_index:i + 1] = np.arange(i + 1 - start_index)
            start_index = i + 1

        return timesteps

    def create_dataset(self):
        def update_inner_state(_transition_index: int, _buffer_num: int, _buffer_loader_dataset: BufferLoaderDataset):
            self.done_idxs.append(len(self.obss) + len(_buffer_loader_dataset.buffer_obss))  # must be placed first
            self.obss.extend(_buffer_loader_dataset.buffer_obss)
            self.actions.extend(_buffer_loader_dataset.buffer_actions)
            self.stepwise_returns.extend(_buffer_loader_dataset.buffer_stepwise_returns)
            self.returns.append(_buffer_loader_dataset.buffer_return)
            self.feedbacks.extend(_buffer_loader_dataset.buffer_feedbacks)
            self.stepwise_auxiliary_feedbacks.extend(_buffer_loader_dataset.buffer_stepwise_auxiliary_feedbacks)
            self.stepwise_ims_largest_smallest_actions_rtg_differences.extend(_buffer_loader_dataset.buffer_stepwise_ims_largest_smallest_actions_rtg_differences)
            self.is_step_allowed_for_feedback.extend(_buffer_loader_dataset.buffer_is_step_allowed_for_feedback)

            self.state_action_feedback_cache.update(_buffer_loader_dataset.buffer_state_action_feedback_cache)
            self.augmented_feedbacks_counters.append(_buffer_loader_dataset.buffer_augmented_feedbacks_counter)
            self.augmented_positive_feedbacks_counters.append(_buffer_loader_dataset.buffer_augmented_positive_feedbacks_counter)
            self.transitions_per_buffer[_buffer_num] += _buffer_loader_dataset.trajectory_step_num
            self.transition_index_per_buffer[_buffer_num] = _transition_index
            self.num_trajectories += 1

        num_of_positive_feedback_equal_to_estimate_large_action = 0
        num_of_negative_feedback_equal_to_estimate_large_action = 0

        if self.states_for_feedbacks_based_on_important_states_filename:
            assert self.num_of_important_states > 0
            file_name = f"{self.num_steps}_{self.game}_{self.seed}_{self.states_for_feedbacks_based_on_important_states_filename}"

            loaded_important_states_indices = np.load(file_name, allow_pickle=True)
            important_states_indices = loaded_important_states_indices['ims_indices'][:self.num_of_important_states]
            important_states_indices_set = set(important_states_indices)

            ims_actions_rtg_differences = loaded_important_states_indices['ims_actions_rtg_differences'][:self.num_of_important_states]

            print(f'important states indices have been loaded, size is {len(important_states_indices_set)}', flush=True)

        while len(self.obss) < self.num_steps:
            buffer_num = self.rng.choice(np.arange(50 - self.num_buffers, 50), 1)[0]
            num_transitions_for_current_buffer = self.transitions_per_buffer[buffer_num]
            transition_index = self.transition_index_per_buffer[buffer_num]

            print(f'loading from buffer {buffer_num} which has {num_transitions_for_current_buffer} already loaded transitions', flush=True)

            replay_capacity = 150000 if (self.game.lower() == 'pong' and self.seed == 512) else 100000
            frb = self._create_replay_buffer(buffer_num, replay_capacity=replay_capacity)
            transitions_loaded = 0

            if frb._loaded_buffers:
                trajectories_to_load = self.trajectories_per_buffer
                buffer_loader_dataset = BufferLoaderDataset()
                feedback_validator = create_feedback_validator(self.game,
                                                               self.augment_only_sparse_reward_with_synthetic_oracle_feedback)
                while transition_index < replay_capacity:
                    if self.feedbacks_enabled:
                        state, ac, ret, next_states, next_action, next_reward, terminal, indices, feedback = frb.sample_transition_batch(
                            batch_size=1, indices=[transition_index])
                    else:
                        state, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(
                            batch_size=1, indices=[transition_index])

                    state = state.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                    buffer_loader_dataset.buffer_obss += [state]
                    buffer_loader_dataset.buffer_actions += [ac[0]]

                    # update inner state of feedback_validator
                    if self.feedbacks_enabled:
                        feedback_validator.handle_step(state, ac[0], ret[0], terminal[0],
                                                                      next_states.transpose((0, 3, 1, 2))[0],
                                                                      next_action[0], len(buffer_loader_dataset.buffer_obss))
                        # save True/False for all states if a state is valid for getting feedback
                        buffer_loader_dataset.buffer_is_step_allowed_for_feedback += [feedback_validator.is_feedback_allow()]

                    # gather all feedbacks
                    if self.feedbacks_enabled:
                        buffer_loader_dataset.buffer_feedbacks += [feedback[0]]
                        buffer_loader_dataset.buffer_stepwise_auxiliary_feedbacks += [-1]
                        buffer_loader_dataset.buffer_stepwise_ims_largest_smallest_actions_rtg_differences += [None]

                    if self.augment_reward_with_synthetic_oracle_feedback_prob and self.rng_augment_reward_with_synthetic_oracle_feedback.rand(1) <= self.augment_reward_with_synthetic_oracle_feedback_prob and feedback_validator.is_feedback_allow():
                        if not self.subset_augment_reward_with_synthetic_oracle_feedback_prob or (self.subset_augment_reward_with_synthetic_oracle_feedback_prob and (1 - self.subset_rng_augment_reward_with_synthetic_oracle_feedback.rand(1)) <= self.subset_augment_reward_with_synthetic_oracle_feedback_prob):
                            if (self.augment_only_sparse_reward_with_synthetic_oracle_feedback and ret[0] == 0) or not self.augment_only_sparse_reward_with_synthetic_oracle_feedback:
                                if (buffer_loader_dataset.buffer_augmented_feedbacks_counter + sum(self.augmented_feedbacks_counters)) < 5000:
                                    # if (feedback[0] > 0 and buffer_loader_dataset.buffer_augmented_positive_feedbacks_counter + sum(self.augmented_positive_feedbacks_counters) < seed_to_number_of_positive_feedbacks_mapping[self.game.lower()][self.seed]) or \
                                    #    (feedback[0] == 0 and ((buffer_loader_dataset.buffer_augmented_feedbacks_counter + sum(self.augmented_feedbacks_counters)) - (buffer_loader_dataset.buffer_augmented_positive_feedbacks_counter + sum(self.augmented_positive_feedbacks_counters))) < 5000 - seed_to_number_of_positive_feedbacks_mapping[self.game.lower()][self.seed]):
                                    buffer_loader_dataset.buffer_stepwise_auxiliary_feedbacks[-1] = feedback[0]

                                    buffer_loader_dataset.buffer_augmented_feedbacks_counter += 1
                                    if feedback[0] > 0:
                                        buffer_loader_dataset.buffer_augmented_positive_feedbacks_counter += 1

                                    state_index = len(self.obss) + len(buffer_loader_dataset.buffer_obss) - 1
                                    buffer_loader_dataset.buffer_state_action_feedback_cache[buffer_loader_dataset.buffer_obss[-1].tobytes()] = (ac[0], feedback[0], state_index, ret[0])

                    if self.states_for_feedbacks_based_on_important_states_filename: # and feedback[0] < 1: # and (buffer_loader_dataset.buffer_augmented_feedbacks_counter + sum(self.augmented_feedbacks_counters)) < 5000:
                        state_index = len(self.obss) + len(buffer_loader_dataset.buffer_obss) - 1
                        if state_index in important_states_indices_set:
                            assert ret[0] == 0 # important states are gathered with states without a reward

                            state_position_in_list = np.where(important_states_indices == state_index)[0][0]
                            largest_and_smallest_rtg_actions = ims_actions_rtg_differences[state_position_in_list][0].split("_")
                            action_with_largest_rtg = int(largest_and_smallest_rtg_actions[0])
                            action_with_smallest_rtg = int(largest_and_smallest_rtg_actions[1])
                            buffer_loader_dataset.buffer_stepwise_auxiliary_feedbacks[-1] = feedback[0]
                            buffer_loader_dataset.buffer_stepwise_ims_largest_smallest_actions_rtg_differences[-1] = (action_with_largest_rtg, action_with_smallest_rtg)

                            if feedback[0] and (ac[0] == action_with_largest_rtg or (ac[0] == 0 and action_with_largest_rtg == 1) or (ac[0] == 1 and action_with_largest_rtg == 0)):
                                num_of_positive_feedback_equal_to_estimate_large_action += 1
                            if feedback[0] == 0 and ac[0] == action_with_largest_rtg:
                                num_of_negative_feedback_equal_to_estimate_large_action += 1

                            buffer_loader_dataset.buffer_augmented_feedbacks_counter += 1
                            if feedback[0] > 0:
                                buffer_loader_dataset.buffer_augmented_positive_feedbacks_counter += 1

                            buffer_loader_dataset.buffer_state_action_feedback_cache[buffer_loader_dataset.buffer_obss[-1].tobytes()] = (ac[0], feedback[0], state_index, ret[0])

                    buffer_loader_dataset.buffer_stepwise_returns += [ret[0]]
                    buffer_loader_dataset.buffer_return += buffer_loader_dataset.buffer_stepwise_returns[-1]

                    transition_index += 1
                    buffer_loader_dataset.trajectory_step_num += 1

                    if terminal[0]:
                        if transition_index >= replay_capacity:
                            self.transition_index_per_buffer[buffer_num] = transition_index
                            break

                        if buffer_loader_dataset.trajectory_step_num < self.context_length:
                           buffer_loader_dataset = BufferLoaderDataset()
                           continue

                        # update global
                        update_inner_state(transition_index, buffer_num, buffer_loader_dataset)
                        transitions_loaded += buffer_loader_dataset.trajectory_step_num

                        trajectories_to_load -= 1
                        if trajectories_to_load == 0:
                            break
                        else:
                            buffer_loader_dataset = BufferLoaderDataset()

            print(f'buffer {buffer_num} has {transitions_loaded} loaded transitions and there are now {len(self.obss)} transitions total divided into {self.num_trajectories} trajectories', flush=True)

        if self.feedbacks_enabled:
            self.num_of_augmented_transitions_with_feedback = sum(self.augmented_feedbacks_counters)
            self.num_of_augmented_transitions_with_positive_feedback = sum(self.augmented_positive_feedbacks_counters)

        actions = np.array(self.actions)
        returns = np.array(self.returns)
        stepwise_returns = np.array(self.stepwise_returns)
        done_idxs = np.array(self.done_idxs)
        feedbacks = np.array(self.feedbacks)
        stepwise_auxiliary_feedbacks = np.array(self.stepwise_auxiliary_feedbacks)
        is_step_allowed_for_feedback = np.array(self.is_step_allowed_for_feedback)
        stepwise_ims_largest_smallest_actions_rtg_differences = np.array(self.stepwise_ims_largest_smallest_actions_rtg_differences)

        # -- create reward-to-go dataset
        rtg = self._create_reward_to_go()
        # -- create timestep dataset
        timesteps = self._create_timestep()

        max_rtg = max(rtg)
        max_timesteps = max(timesteps)
        transitions_per_buffer_num = dict(enumerate(self.transitions_per_buffer.flatten(), 1)).__str__()

        print(f'max rtg is {max_rtg}', flush=True)
        print(f'trajectory quality is {rtg.mean()}', flush=True)
        print(f'max timestep is {max_timesteps}', flush=True)
        print_number_of_states_with_multiple_actions(self.obss, actions)

        if self.feedbacks_enabled:
            print(f"num of augmented rewards with feedback is {self.num_of_augmented_transitions_with_feedback}")
            print(f"num of augmented rewards with positive feedback is {self.num_of_augmented_transitions_with_positive_feedback}")
            print(f"num of augmented implicit feedback is {self.num_of_augmented_implicit_feedbacks}")
            print(f"num of augmented implicit positive feedback is {self.num_of_augmented_implicit_with_positive_feedbacks}")

            if self.verbose > 0:
                store_to_file_augmented_states_with_oracle_action(self.game, self.state_action_feedback_cache)

        state_action_positive_feedback_cache_size = len(self.state_action_positive_feedback_cache)
        print(f"state_action_positive_feedback_cache size: {state_action_positive_feedback_cache_size}")

        print_statistics(actions, feedbacks, self.state_action_positive_feedback_cache,
                         returns, stepwise_returns, rtg, done_idxs)

        # store dataset statistics
        dataset_statistics = DatasetStatistics(len(self.obss),
                                               self.num_trajectories,
                                               max_rtg,
                                               max_timesteps,
                                               transitions_per_buffer_num,
                                               self.num_of_augmented_transitions_with_feedback,
                                               self.num_of_augmented_transitions_with_positive_feedback,
                                               self.num_of_augmented_implicit_feedbacks,
                                               self.num_of_augmented_implicit_with_positive_feedbacks,
                                               state_action_positive_feedback_cache_size)

        print("create dataset process has finished")

        print(num_of_positive_feedback_equal_to_estimate_large_action)
        print(num_of_negative_feedback_equal_to_estimate_large_action)

        return self.obss, self.actions, self.returns, self.done_idxs, rtg, timesteps, self.feedbacks, self.state_action_positive_feedback_cache, \
               stepwise_auxiliary_feedbacks, is_step_allowed_for_feedback, stepwise_ims_largest_smallest_actions_rtg_differences, dataset_statistics
