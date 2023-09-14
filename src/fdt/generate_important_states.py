import argparse
import concurrent
import hashlib
import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List

import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
import torch.multiprocessing as mp
from torch import nn

from utils import set_seed
from feedback_validator import BreakoutFeedbackValidator, \
    PongFeedbackValidator, QbertFeedbackValidator, SeaquestFeedbackValidator
from fixed_replay_buffer import FixedReplayBuffer


def max_pool_hash(state: np.array, num_frames: int, device: str = "cpu", kernel_size: int = 7, stride: int = 7):
    max_pool_operation = nn.MaxPool2d(kernel_size, stride=stride)
    state_as_tensor = torch.tensor(state, dtype=torch.float32, device=device)
    state_as_tensor = max_pool_operation(state_as_tensor)
    if device == "cpu":
        m = hashlib.sha512(state_as_tensor[4 - num_frames:].flatten().numpy())
    else:
        m = hashlib.sha512(state_as_tensor[4 - num_frames:].flatten().cpu().numpy())

    return m


def print_number_of_states_with_multiple_actions(states, actions):
    state_to_actions = defaultdict(set)

    for i, state in enumerate(states):
        action = actions[i]
        state_to_actions[state].add(action)

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
    print(
        f"states actions coverage is {number_of_states_with_1_actions + number_of_states_with_2_actions + number_of_states_with_3_actions + number_of_states_with_4_actions_or_more}")


def calc_important_states(game, seed, num_steps, num_frames, gamma,
                          kernel_size, stride,
                          num_trajectories, states, states_images, actions,
                          rtgs, is_step_allowed_for_feedback, done_idxs,
                          minimum_num_state_rtgs_to_consider_for_important_state,
                          window_size_for_important_state,
                          notify_important_states_with_multiple_occurrences_n,
                          first_visit_state_action: bool = True,
                          life_loss_negative_reward: int = 0,
                          steps_ahead_for_terminal_or_life_loss_state: int = 0
                          ):
    state_to_actions_to_rtg = {}
    states_to_index_and_trajectory = {}
    state_to_actions_with_max_differences_of_rtgs = {}
    file_lines = []
    valid_idx_range = 500000

    print("Calculating important states process has started")

    if len(is_step_allowed_for_feedback) == 0:
        return

    def convert_states_to_actions_rtg():
        trajectory = 0
        done_idxs_i = 0
        state_to_actions_coming_from_invalid_states = defaultdict(set)

        num_states_that_sometimes_not_valid_but_already_candidates = 0
        num_states_that_are_skipped = 0

        state_action_pairs_in_trajectory = set()
        for i, state in enumerate(states):
            if not is_step_allowed_for_feedback[i]:
                if state in state_to_actions_to_rtg:
                    num_states_that_sometimes_not_valid_but_already_candidates += 1
                    action = actions[i]
                    rtg = rtgs[i]

                    if action not in state_to_actions_to_rtg[state]: # utilize actions in invalid states that are already candidates for feedback
                        state_to_actions_to_rtg[state][action] = [rtg]
                        state_to_actions_coming_from_invalid_states[state].add(action)
                    elif action in state_to_actions_coming_from_invalid_states[state]: # only append rtg to actions associate to invalid states
                        state_to_actions_to_rtg[state][action].append(rtg) # TODO add this logic here as well: if not first_visit_state_action or (first_visit_state_action and not f"{state}_{action}" in state_action_pairs_in_trajectory):
                if i + 1 == done_idxs[done_idxs_i]:
                    trajectory += 1
                    done_idxs_i += 1
                    state_action_pairs_in_trajectory = set()
                continue

            action = actions[i]
            rtg = rtgs[i]
            if i <= valid_idx_range:
                if state not in states_to_index_and_trajectory:
                    states_to_index_and_trajectory[state] = [(i, trajectory, action)]  # store mapping between state <-> first index, trajectory num, action
                else:
                    states_to_index_and_trajectory[state].append((i, trajectory, action))

            if state not in state_to_actions_to_rtg:
                if i <= valid_idx_range:
                    state_to_actions_to_rtg[state] = {action: [rtg]}
                else:
                    num_states_that_are_skipped += 1
            else:
                if action not in state_to_actions_to_rtg[state]:
                    state_to_actions_to_rtg[state][action] = [rtg]
                else:
                    if not first_visit_state_action or (first_visit_state_action and not f"{state}_{action}" in state_action_pairs_in_trajectory):
                        state_to_actions_to_rtg[state][action].append(rtg)

            state_action_pairs_in_trajectory.add(f"{state}_{action}")

            if i + 1 == done_idxs[done_idxs_i]:
                trajectory += 1
                done_idxs_i += 1
                state_action_pairs_in_trajectory = set()

        print(f"num_states_that_sometimes_not_valid_but_already_candidates: {num_states_that_sometimes_not_valid_but_already_candidates}", flush=True)
        file_lines.append(f"num_states_that_sometimes_not_valid_but_already_candidates {num_states_that_sometimes_not_valid_but_already_candidates}\n")
        print(f"num_states_that_are_skipped: {num_states_that_are_skipped}", flush=True)
        file_lines.append(f"num_states_that_are_skipped {num_states_that_are_skipped}\n")

    def calc_avg_of_states_to_actions_rtg():
        state_to_actions_to_remove = {}

        # convert to numpy array and take avg
        for state, actions_to_rtg in state_to_actions_to_rtg.items():
            for action, rtg_list in actions_to_rtg.items():
                actions_to_rtg[action] = np.asarray(rtg_list).mean()

                if minimum_num_state_rtgs_to_consider_for_important_state > 0:
                    if len(rtg_list) < minimum_num_state_rtgs_to_consider_for_important_state:
                        if state not in state_to_actions_to_remove:
                            state_to_actions_to_remove[state] = [action]
                        else:
                            state_to_actions_to_remove[state].append(action)

        if minimum_num_state_rtgs_to_consider_for_important_state > 0:
            num_of_states_with_more_than_2_actions_before_removing_actions = 0
            num_of_states_with_more_than_2_actions_after_removing_actions = 0
            for state, actions_list in state_to_actions_to_remove.items():
                if len(state_to_actions_to_rtg[state]) >= 2:
                    num_of_states_with_more_than_2_actions_before_removing_actions += 1
                for action in actions_list:
                    del state_to_actions_to_rtg[state][action]
                if len(state_to_actions_to_rtg[state]) >= 2:
                    num_of_states_with_more_than_2_actions_after_removing_actions += 1

            num_of_states_ignored = num_of_states_with_more_than_2_actions_before_removing_actions - num_of_states_with_more_than_2_actions_after_removing_actions
            print(f"num_of_state_ignored_due_to_small_rtg_samples: {num_of_states_ignored}", flush=True)
            file_lines.append(f"num_of_state_ignored_due_to_small_rtg_samples {num_of_states_ignored}\n")
            print(f"num_of_states_with_more_than_2_actions_before_removing_actions: {num_of_states_with_more_than_2_actions_before_removing_actions}", flush=True)
            file_lines.append(f"num_of_states_with_more_than_2_actions_before_removing_actions {num_of_states_with_more_than_2_actions_before_removing_actions}\n")

    def calc_state_to_max_differences_of_rtgs():
        num_states_with_1_action = 0
        for state, actions_to_rtg in state_to_actions_to_rtg.items():
            if len(actions_to_rtg) >= 2:
                action_with_largest_rtg = max(actions_to_rtg, key=actions_to_rtg.get)
                action_with_smallest_rtg = min(actions_to_rtg, key=actions_to_rtg.get)

                rtg_diff = actions_to_rtg[action_with_largest_rtg] - actions_to_rtg[action_with_smallest_rtg]
                if rtg_diff > 0:
                    state_to_actions_with_max_differences_of_rtgs[state] = (f"{action_with_largest_rtg}_{action_with_smallest_rtg}",
                                                               actions_to_rtg[action_with_largest_rtg] - actions_to_rtg[action_with_smallest_rtg])
            else:
                num_states_with_1_action += 1

        print(f"num of valid states with 1 action is {num_states_with_1_action}", flush=True)
        print(f"num of valid states with more than 2 actions is {len(state_to_actions_with_max_differences_of_rtgs)}", flush=True)
        file_lines.append(f"num of valid states with more than 2 actions is {len(state_to_actions_with_max_differences_of_rtgs)}\n")

        # Using sorted() + itemgetter() + items()
        return sorted(state_to_actions_with_max_differences_of_rtgs.items(), key=lambda x: state_to_actions_with_max_differences_of_rtgs[x[0]][1], reverse=True)

    def collect_important_states_indices(state_to_actions_with_max_differences_of_rtgs_sorted):
        _important_states_indices = []
        _important_states_actions_rtg_differences = []
        num_of_added_states_due_to_multiple_occurrences = 0
        for i, (state, (actions, rtg_diff)) in enumerate(state_to_actions_with_max_differences_of_rtgs_sorted):
            largest_and_smallest_rtg_actions = actions.split("_")
            action_with_largest_rtg = int(largest_and_smallest_rtg_actions[0])
            action_with_smallest_rtg = int(largest_and_smallest_rtg_actions[1])

            state_indices = []
            for state_index, _, action in states_to_index_and_trajectory[state]:
                if action == action_with_largest_rtg:
                    _important_states_indices.append(state_index)
                    _important_states_actions_rtg_differences.append((actions, rtg_diff, action_with_largest_rtg))
                    state_indices.append(state_index)
                    break

            for state_index, _, action in states_to_index_and_trajectory[state]:
                if action == action_with_smallest_rtg:
                    _important_states_indices.append(state_index)
                    _important_states_actions_rtg_differences.append((actions, rtg_diff, action_with_smallest_rtg))
                    state_indices.append(state_index)
                    break

            if len(state_indices) == 0:
                state_index = states_to_index_and_trajectory[state][0][0]
                _important_states_indices.append(state_index)
                _important_states_actions_rtg_differences.append((actions, rtg_diff, states_to_index_and_trajectory[state][0][2]))
                state_indices.append(state_index)

            initial_state_indices_size = len(state_indices)
            if notify_important_states_with_multiple_occurrences_n > 0:
                for state_index, _, action in states_to_index_and_trajectory[state]:
                    if len(state_indices) - initial_state_indices_size > notify_important_states_with_multiple_occurrences_n:
                        break

                    if state_index in state_indices:
                        continue

                    _important_states_indices.append(state_index)
                    _important_states_actions_rtg_differences.append((actions, rtg_diff, action))
                    state_indices.append(state_index)
                    num_of_added_states_due_to_multiple_occurrences += 1

                    # # favor action with largest rtg TODO
                    # if action == action_with_smallest_rtg:
                    #     _important_states_indices.append(state_index)
                    #     _important_states_actions_rtg_differences.append((actions, rtg_diff))
                    #     state_indices.append(state_index)
                    #     num_of_added_states_due_to_multiple_occurrences += 1
                    #     continue
                    #
                    # if action == action_with_smallest_rtg:
                    #     _important_states_indices.append(state_index)
                    #     _important_states_actions_rtg_differences.append((actions, rtg_diff))
                    #     state_indices.append(state_index)
                    #     num_of_added_states_due_to_multiple_occurrences += 1
                    #     continue

        if notify_important_states_with_multiple_occurrences_n > 0:
            print(f"num of added states due to multiple occurrences {num_of_added_states_due_to_multiple_occurrences}",
                  flush=True)
            file_lines.append(f"num of added states due to multiple occurrences {num_of_added_states_due_to_multiple_occurrences}\n")

        if window_size_for_important_state > 0:
            ims_indices_to_remove = set()
            for i, ims_idx in enumerate(_important_states_indices):
                for j, ims_idx_inner in enumerate(_important_states_indices):
                    if ims_idx == ims_idx_inner:
                        continue

                    if abs(ims_idx - ims_idx_inner) <= window_size_for_important_state and i < j:
                        ims_indices_to_remove.add(j)

            _important_states_indices = [idx for i, idx in enumerate(_important_states_indices) if i not in ims_indices_to_remove]
            _important_states_actions_rtg_differences = [(actions, rtg_diff, action) for i, (actions, rtg_diff, action) in enumerate(_important_states_actions_rtg_differences) if i not in ims_indices_to_remove]

            num_of_important_state_ignored_due_to_window_size = len(ims_indices_to_remove)
            print(f"num of important states that were ignored due to window size {num_of_important_state_ignored_due_to_window_size}", flush=True)
            file_lines.append(f"num of important states that were ignored due to window size {num_of_important_state_ignored_due_to_window_size}\n")

        return _important_states_indices, _important_states_actions_rtg_differences

    file_lines.append(f"{len(states)} transitions total divided into {num_trajectories} trajectories\n")

    total_num_states_allowed_for_feedback = np.count_nonzero(is_step_allowed_for_feedback)
    print(f"total num of states allowed for feedback is {total_num_states_allowed_for_feedback}", flush=True)
    file_lines.append(f"total num of states allowed for feedback is {total_num_states_allowed_for_feedback}\n")
    num_states_allowed_for_feedback = np.count_nonzero(is_step_allowed_for_feedback[:valid_idx_range])
    print(f"num of states allowed for feedback is {num_states_allowed_for_feedback}", flush=True)
    file_lines.append(f"num of states allowed for feedback is {num_states_allowed_for_feedback}\n")

    convert_states_to_actions_rtg()
    print(f"num of valid states is {len(state_to_actions_to_rtg)}", flush=True)
    file_lines.append(f"num of valid states is {len(state_to_actions_to_rtg)}\n")

    calc_avg_of_states_to_actions_rtg()

    state_to_actions_with_max_differences_of_rtgs_sorted = calc_state_to_max_differences_of_rtgs()

    # collect important states and their indices & trajectory
    important_states_indices, important_states_actions_rtg_differences = collect_important_states_indices(state_to_actions_with_max_differences_of_rtgs_sorted)

    print(f"important states num: {len(important_states_indices)}", flush=True)
    print(f"important states indices: {important_states_indices}", flush=True)
    print(f"important states actions rtg differences: {important_states_actions_rtg_differences}", flush=True)
    file_lines.append(f"important states num: {len(important_states_indices)}\n")
    file_lines.append(f"important states indices: {important_states_indices}\n")
    file_lines.append(f"important states actions rtg differences: {important_states_actions_rtg_differences}\n")

    file_name = generate_ims_file_name(game,
                                       seed,
                                       num_steps,
                                       num_frames,
                                       gamma,
                                       kernel_size,
                                       stride,
                                       minimum_num_state_rtgs_to_consider_for_important_state,
                                       window_size_for_important_state,
                                       notify_important_states_with_multiple_occurrences_n,
                                       first_visit_state_action,
                                       life_loss_negative_reward,
                                       steps_ahead_for_terminal_or_life_loss_state
    )
    # Writing to file
    with open(f"{file_name}.txt", "w") as file:
        # Writing data to a file
        file.writelines(file_lines)

    np.savez_compressed(file_name,
                        ims_indices=np.array(important_states_indices),
                        ims_actions_rtg_differences=np.array(important_states_actions_rtg_differences),
                        states=np.array(states_images))


def create_feedback_validator(game: str,
                              augment_only_sparse_reward_with_synthetic_oracle_feedback: bool):
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
    buffer_obss: list = field(default_factory=lambda: [])
    buffer_states: list = field(default_factory=lambda: [])
    buffer_actions: list = field(default_factory=lambda: [])
    buffer_return: int = field(default=0)
    buffer_feedbacks: list = field(default_factory=lambda: [])
    buffer_stepwise_auxiliary_feedbacks: list = field(default_factory=lambda: [])
    buffer_is_step_allowed_for_feedback: list = field(default_factory=lambda: [])
    buffer_stepwise_returns: list = field(default_factory=lambda: [])
    buffer_stepwise_terminals: list = field(default_factory=lambda: [])
    trajectory_step_num: int = field(default=0)
    buffer_state_action_feedback_cache: dict = field(default_factory=lambda: {})
    buffer_augmented_feedbacks_counter: int = field(default=0)
    buffer_augmented_positive_feedbacks_counter: int = field(default=0)


class ImportantStatesCreator:
    def __init__(self,
                 num_buffers: int,
                 num_steps: int,
                 num_frames: int,
                 game: str,
                 data_dir_prefix: str,
                 trajectories_per_buffer: int,
                 context_length: int,
                 augment_only_sparse_reward_with_synthetic_oracle_feedback: bool = False,
                 gamma: float = 0.9,
                 kernel_size: int = 7,
                 stride: int = 7,
                 run_in_parallel: int = 0,
                 seed: int = 123,
                 minimum_num_state_rtgs_to_consider_for_important_state: int = 0,
                 window_size_for_important_state: int = 0,
                 notify_important_states_with_multiple_occurrences_n: bool = False,
                 first_visit_state_action: bool = True,
                 device: str = "cpu",
                 life_loss_negative_reward: int = 0,
                 steps_ahead_for_terminal_or_life_loss_state: int = 0,
                 verbose: int = 0):
        print(
            f"create dataset process with num_buffers={num_buffers}, num_steps={num_steps}, game={game}, data_dir_prefix={data_dir_prefix}, trajectories_per_buffer={trajectories_per_buffer}, "
            f"context_length={context_length}, gamma={gamma}, seed={seed} has started", flush=True)

        self.device = device
        self.run_in_parallel = run_in_parallel
        self.num_buffers = num_buffers
        self.num_steps = num_steps
        self.num_frames = num_frames
        self.game = game
        self.data_dir_prefix = data_dir_prefix
        self.trajectories_per_buffer = trajectories_per_buffer
        self.context_length = context_length
        self.augment_only_sparse_reward_with_synthetic_oracle_feedback = augment_only_sparse_reward_with_synthetic_oracle_feedback
        self.gamma = gamma
        self.kernel_size = kernel_size
        self.stride = stride
        self.seed = seed
        self.minimum_num_state_rtgs_to_consider_for_important_state = minimum_num_state_rtgs_to_consider_for_important_state
        self.window_size_for_important_state = window_size_for_important_state
        self.notify_important_states_with_multiple_occurrences_n = notify_important_states_with_multiple_occurrences_n
        self.first_visit_state_action = first_visit_state_action
        self.life_loss_negative_reward = life_loss_negative_reward
        self.steps_ahead_for_terminal_or_life_loss_state = steps_ahead_for_terminal_or_life_loss_state
        self.verbose = verbose

        self.rng = np.random.RandomState(seed)

        self.obss = []
        self.states = []
        self.actions = []
        self.returns = []
        self.feedbacks = []
        self.stepwise_auxiliary_feedbacks = []
        self.is_step_allowed_for_feedback = []
        self.state_action_feedback_cache = {}
        self.done_idxs = []
        self.stepwise_returns = []
        self.stepwise_terminals = []
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

    @staticmethod
    def create_replay_buffer(game: str, data_dir_prefix: str, buffer_num: int, agent_dir_number: str = '1'):
        data_dir = os.sep.join([data_dir_prefix, game, agent_dir_number, 'replay_logs'])
        frb = FixedReplayBuffer(
            data_dir=data_dir,
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=1000000,
            extra_storage_types=None)

        return frb

    def _create_reward_to_go(self):
        # -- create reward-to-go dataset
        start_index = 0
        rtg = np.zeros_like(self.stepwise_returns)
        gamma = self.gamma

        for i in self.done_idxs:
            i = int(i)
            curr_trajectory_returns = self.stepwise_returns[start_index:i]
            curr_trajectory_terminals = self.stepwise_terminals[start_index:i]

            for j in range(i - 1, start_index - 1, -1):  # start from i-1
                rtg_j = curr_trajectory_returns[j - start_index:i - start_index]
                cumulative_discount_vector = np.array([math.pow(gamma, n) for n in range(len(rtg_j))],
                                                      dtype=np.float32)
                rtg[j] = np.sum(cumulative_discount_vector * rtg_j, axis=0)

                if self.game.lower() == 'breakout':
                    terminals_j = curr_trajectory_terminals[j - start_index:i - start_index][:self.steps_ahead_for_terminal_or_life_loss_state]
                elif self.game.lower() == 'pong':
                    terminals_j = curr_trajectory_terminals[j - start_index:i - start_index][6:self.steps_ahead_for_terminal_or_life_loss_state]
                elif self.game.lower() == 'qbert':
                    terminals_j = curr_trajectory_terminals[j - start_index:i - start_index][:self.steps_ahead_for_terminal_or_life_loss_state]
                elif self.game.lower() == 'seaquest':
                    terminals_j = curr_trajectory_terminals[j - start_index:i - start_index][9:self.steps_ahead_for_terminal_or_life_loss_state]

                if len(rtg_j) > self.steps_ahead_for_terminal_or_life_loss_state:
                    if np.any(terminals_j):
                        rtg[j] += self.life_loss_negative_reward
                else:
                    rtg[j] += self.life_loss_negative_reward

            start_index = i

        return rtg

    @staticmethod
    def collect_trajectories(data_dir_prefix: str,
                             game: str,
                             obss_size: int,
                             num_steps: int,
                             num_transitions_for_current_buffer: int,
                             transition_index: int,
                             trajectories_per_buffer: int,
                             augment_only_sparse_reward_with_synthetic_oracle_feedback: bool,
                             num_frames: int,
                             context_length: int,
                             agent_dir_number: int,
                             buffer_num: int,
                             change_trajectories_per_buffer: int,
                             device: str,
                             kernel_size: int,
                             stride: int) -> Tuple[int, int, int, int, List[BufferLoaderDataset]]:
        if obss_size >= num_steps:
            raise Exception()

        buffer_loader_dataset = BufferLoaderDataset()
        buffer_loader_datasets = []
        transitions_loaded = 0

        print(
            f'loading from agent dir {agent_dir_number} buffer {buffer_num} which has {num_transitions_for_current_buffer} already loaded transitions',
            flush=True)

        dataset_size = 999999
        if transition_index >= dataset_size:
            print(f'agent dir {agent_dir_number} buffer {buffer_num} has {transitions_loaded} loaded transitions', flush=True)
            return agent_dir_number, buffer_num, transition_index, transitions_loaded, buffer_loader_datasets

        frb = ImportantStatesCreator.create_replay_buffer(game, data_dir_prefix, buffer_num, agent_dir_number=f"{agent_dir_number}")

        if frb._loaded_buffers:
            trajectories_to_load = trajectories_per_buffer
            feedback_validator = create_feedback_validator(game,
                                                           augment_only_sparse_reward_with_synthetic_oracle_feedback)
            while transition_index < dataset_size:
                state, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(
                    batch_size=1, indices=[transition_index])

                state = state.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)

                if game.lower() == 'breakout':
                    #m = hashlib.sha512(state[4 - num_frames:, 9:, :].flatten()) # original
                    state_temp = np.copy(state)
                    state_temp[:, :6, :48] = 0 # without score

                    m = max_pool_hash(state_temp, num_frames, device, kernel_size=kernel_size, stride=stride)
                elif game.lower() == 'qbert':
                    state_temp = np.copy(state)
                    state_temp[:, 2:12, 17:38] = 0
                    m = max_pool_hash(state_temp, num_frames, device, kernel_size=kernel_size, stride=stride)
                elif game.lower() == 'pong':
                    state_temp = np.copy(state)
                    state_temp = state_temp[:, 11:, :]
                    m = max_pool_hash(state_temp, num_frames, device, kernel_size=kernel_size, stride=stride)
                else:  # seaquest
                    state_temp = np.copy(state)
                    state_temp[:, 18:22, :][state_temp[:, 18:22, :] < 100] = 0 # emphasize ship shape
                    state_temp[:, 18:20, :] = 0  # emphasize ship shape
                    state_temp[:, 21, :] = 0  # emphasize ship shape
                    state_temp[:, 18:22, :][state_temp[:, 18:22, :] >= 100] = 172
                    state_temp[:, 61:70, :] = 0 # remove oxygen area
                    state_temp = state_temp[:, 18:, :] # without score + lives
                    m = max_pool_hash(state_temp, num_frames, device, kernel_size=kernel_size, stride=stride)

                buffer_loader_dataset.buffer_obss += [m.hexdigest()]
                if obss_size < change_trajectories_per_buffer:
                    buffer_loader_dataset.buffer_states += [state]

                buffer_loader_dataset.buffer_actions += [ac[0]]

                # update inner state of feedback_validator
                feedback_validator.handle_step(state, ac[0], ret[0], terminal[0],
                                                              next_states.transpose((0, 3, 1, 2))[0],
                                                              next_action[0],
                                                              len(buffer_loader_dataset.buffer_obss))
                # save True/False for all states if a state is valid for getting feedback
                enable_sticky_action_logic = obss_size < change_trajectories_per_buffer
                buffer_loader_dataset.buffer_is_step_allowed_for_feedback += [
                    feedback_validator.is_feedback_allow(enable_sticky_action_logic)]

                buffer_loader_dataset.buffer_stepwise_returns += [ret[0]]
                buffer_loader_dataset.buffer_return += buffer_loader_dataset.buffer_stepwise_returns[-1]

                is_life_loss = feedback_validator.is_life_loss or terminal[0]
                buffer_loader_dataset.buffer_stepwise_terminals += [is_life_loss]

                transition_index += 1
                buffer_loader_dataset.trajectory_step_num += 1

                if terminal[0]:
                    if transition_index >= dataset_size:
                        break

                    if buffer_loader_dataset.trajectory_step_num < context_length:
                        buffer_loader_dataset = BufferLoaderDataset()
                        continue

                    # update global
                    buffer_loader_datasets.append(buffer_loader_dataset)
                    transitions_loaded += buffer_loader_dataset.trajectory_step_num

                    trajectories_to_load -= 1
                    if trajectories_to_load == 0:
                        break
                    else:
                        buffer_loader_dataset = BufferLoaderDataset()

        print(
            f'agent dir {agent_dir_number} buffer {buffer_num} has {transitions_loaded} loaded transitions',
            flush=True)

        return agent_dir_number, buffer_num, transition_index, transitions_loaded, buffer_loader_datasets

    def generate_important_states(self):
        def update_inner_state(_transition_index: int, _buffer_num: int, _buffer_loader_dataset: BufferLoaderDataset):
            self.done_idxs.append(len(self.obss) + len(_buffer_loader_dataset.buffer_obss))  # must be placed first
            self.obss.extend(_buffer_loader_dataset.buffer_obss)
            self.states.extend(_buffer_loader_dataset.buffer_states)
            self.actions.extend(_buffer_loader_dataset.buffer_actions)
            self.stepwise_returns.extend(_buffer_loader_dataset.buffer_stepwise_returns)
            self.stepwise_terminals.extend(_buffer_loader_dataset.buffer_stepwise_terminals)
            self.returns.append(_buffer_loader_dataset.buffer_return)
            self.feedbacks.extend(_buffer_loader_dataset.buffer_feedbacks)
            self.stepwise_auxiliary_feedbacks.extend(_buffer_loader_dataset.buffer_stepwise_auxiliary_feedbacks)
            self.is_step_allowed_for_feedback.extend(_buffer_loader_dataset.buffer_is_step_allowed_for_feedback)

            self.state_action_feedback_cache.update(_buffer_loader_dataset.buffer_state_action_feedback_cache)
            self.augmented_feedbacks_counters.append(_buffer_loader_dataset.buffer_augmented_feedbacks_counter)
            self.augmented_positive_feedbacks_counters.append(
                _buffer_loader_dataset.buffer_augmented_positive_feedbacks_counter)
            self.transitions_per_buffer[_buffer_num] += _buffer_loader_dataset.trajectory_step_num
            self.transition_index_per_buffer[_buffer_num] = _transition_index
            self.num_trajectories += 1

        change_trajectories_per_buffer = 550000
        first_buffer_incremental = True
        buffer_num = 0
        agent_dir_number = 1
        exit_loop_flag = False
        while len(self.obss) < self.num_steps and not exit_loop_flag:
            if len(self.obss) > change_trajectories_per_buffer:
                self.trajectories_per_buffer = 1500

            if len(self.obss) < change_trajectories_per_buffer:
                buffer_num = self.rng.choice(np.arange(50 - self.num_buffers, 50), 1)[0]
                #buffer_num = self.rng.choice(np.arange(0, self.num_buffers), 1)[0] # suboptimal dataset
            else:
                if self.run_in_parallel > 0:
                    with ProcessPoolExecutor(max_workers=self.run_in_parallel,
                                             mp_context=mp.get_context('spawn')) as executor:
                        while True:
                            tasks = []

                            for buffer_num in range(1, self.num_buffers - 1):
                                if ((agent_dir_number - 1) * 47000000) + buffer_num * 1000000 - 1000000 >= self.num_steps:
                                    break

                                tasks.append((self.data_dir_prefix, self.game, len(self.obss), self.num_steps, self.transitions_per_buffer[buffer_num],
                                              self.transition_index_per_buffer[buffer_num],
                                              self.trajectories_per_buffer, self.augment_only_sparse_reward_with_synthetic_oracle_feedback, self.num_frames, self.context_length,
                                              agent_dir_number, buffer_num, change_trajectories_per_buffer, self.device,
                                              self.kernel_size, self.stride))

                            futures = {executor.submit(self.collect_trajectories, *task): task for task in tasks}

                            print(f"submit {len(tasks)} tasks", flush=True)
                            if len(tasks) == 0:
                                exit_loop_flag = True
                                break

                            results = []
                            for future in concurrent.futures.as_completed(futures):
                                try:
                                    agent_dir_number, buffer_num, transition_index, transitions_loaded, buffer_loader_datasets = future.result()
                                    results.append(
                                        [agent_dir_number, buffer_num, transition_index, transitions_loaded, buffer_loader_datasets])
                                except Exception as e:
                                    print(f"collect trajectories computation has failed with exception: {e}")
                                    raise

                            results = sorted(results, key=lambda x: (x[0], x[1]))

                            for agent_dir_number, buffer_num, transition_index, transitions_loaded, buffer_loader_datasets in results:
                                if len(self.obss) >= self.num_steps:
                                    break

                                # update global
                                for buffer_loader_dataset in buffer_loader_datasets:
                                    update_inner_state(transition_index, buffer_num, buffer_loader_dataset)

                                print(
                                    f'agent dir {agent_dir_number} buffer {buffer_num} has {transitions_loaded} loaded transitions and there are now {len(self.obss)} transitions total divided into {self.num_trajectories} trajectories',
                                    flush=True)

                            if len(self.obss) >= self.num_steps:
                                exit_loop_flag = True
                                break
                            else:
                                agent_dir_number += 1
                                if agent_dir_number > 5:
                                    exit_loop_flag = True
                                    break
                                self.transitions_per_buffer = np.zeros(self.num_buffers, dtype=int)
                                self.transition_index_per_buffer = np.zeros(self.num_buffers, dtype=int)
                else:
                    buffer_num = 1 + buffer_num if not first_buffer_incremental else 1
                    if first_buffer_incremental:
                        first_buffer_incremental = False

                    if len(self.obss) > 47000000 * agent_dir_number:
                        agent_dir_number += 1
                        buffer_num = 1
                        self.transitions_per_buffer = np.zeros(self.num_buffers, dtype=int)
                        self.transition_index_per_buffer = np.zeros(self.num_buffers, dtype=int)

            if self.run_in_parallel == 0 or len(self.obss) < change_trajectories_per_buffer:
                agent_dir_number, buffer_num, transition_index, transitions_loaded, buffer_loader_datasets = \
                    self.collect_trajectories(self.data_dir_prefix, self.game, len(self.obss), self.num_steps,
                                              self.transitions_per_buffer[buffer_num],
                                              self.transition_index_per_buffer[buffer_num],
                                              self.trajectories_per_buffer,
                                              self.augment_only_sparse_reward_with_synthetic_oracle_feedback,
                                              self.num_frames,
                                              self.context_length,
                                              agent_dir_number,
                                              buffer_num,
                                              change_trajectories_per_buffer,
                                              self.device,
                                              self.kernel_size,
                                              self.stride)
                # update global
                for buffer_loader_dataset in buffer_loader_datasets:
                    update_inner_state(transition_index, buffer_num, buffer_loader_dataset)
                print(
                    f'agent dir {agent_dir_number} buffer {buffer_num} has {transitions_loaded} loaded transitions and there are now {len(self.obss)} transitions total divided into {self.num_trajectories} trajectories',
                    flush=True)

        actions = np.array(self.actions)
        is_step_allowed_for_feedback = np.array(self.is_step_allowed_for_feedback)

        # -- create reward-to-go dataset
        rtg = self._create_reward_to_go()

        max_rtg = max(rtg)

        print(f'max rtg is {max_rtg}', flush=True)
        print(f'trajectory quality is {rtg.mean()}', flush=True)

        calc_important_states(self.game, self.seed, self.num_steps, self.num_frames, self.gamma,
                              self.kernel_size, self.stride,
                              self.num_trajectories, self.obss, self.states, actions, rtg,
                              is_step_allowed_for_feedback, self.done_idxs,
                              self.minimum_num_state_rtgs_to_consider_for_important_state,
                              self.window_size_for_important_state,
                              self.notify_important_states_with_multiple_occurrences_n,
                              self.first_visit_state_action,
                              self.life_loss_negative_reward,
                              self.steps_ahead_for_terminal_or_life_loss_state
        )
        print_number_of_states_with_multiple_actions(self.obss, actions)

        print("generate important states process has finished")

        return


def generate_important_states(args):
    important_states_creator: ImportantStatesCreator = ImportantStatesCreator(args.num_buffers,
                                                                              args.num_steps,
                                                                              args.num_frames,
                                                                              args.game,
                                                                              args.data_dir_prefix,
                                                                              args.trajectories_per_buffer,
                                                                              args.context_length,
                                                                              args.augment_only_sparse_reward_with_synthetic_oracle_feedback,
                                                                              args.gamma,
                                                                              args.kernel_size_and_stride[0],
                                                                              args.kernel_size_and_stride[1],
                                                                              args.run_in_parallel,
                                                                              args.generate_dataset_seed,
                                                                              args.minimum_num_state_rtgs_to_consider_for_important_state,
                                                                              args.window_size_for_important_state,
                                                                              args.notify_important_states_with_multiple_occurrences_n,
                                                                              args.first_visit_state_action,
                                                                              args.device,
                                                                              args.life_loss_negative_reward,
                                                                              args.steps_ahead_for_terminal_or_life_loss_state)
    important_states_creator.generate_important_states()


def generate_ims_file_name(game: str,
                           seed: int,
                           num_steps: int,
                           num_frames: int,
                           gamma: float,
                           kernel_size: int,
                           stride: int,
                           minimum_num_state_rtgs_to_consider_for_important_state: int,
                           window_size_for_important_state: int,
                           notify_important_states_with_multiple_occurrences_n: int,
                           first_visit_state_action: bool,
                           life_loss_negative_reward: int,
                           steps_ahead_for_terminal_or_life_loss_state: int) -> str:
    file_name = f"500000_{game}_{seed}_{gamma}_{kernel_size}-{stride}_{num_steps}_{num_frames}_{minimum_num_state_rtgs_to_consider_for_important_state}_{window_size_for_important_state}_multiple_occurrences_{notify_important_states_with_multiple_occurrences_n}_first_visit_{first_visit_state_action}_important_states_indices_with_life_loss_{steps_ahead_for_terminal_or_life_loss_state}_{life_loss_negative_reward}"

    return file_name


def run(args):
    # make deterministic
    set_seed(args.seed)
    seed_everything(args.seed)

    args.game = args.game.strip("'")
    args.data_dir_prefix = args.data_dir_prefix.strip("'")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    print(args, flush=True)

    args.device = "cpu"
    if args.gpu:
        args.device = "cuda:0"

    kernel_size = args.kernel_size_and_stride[0]
    stride = args.kernel_size_and_stride[1]

    file_name = generate_ims_file_name(args.game,
                                       args.seed,
                                       args.num_steps,
                                       args.num_frames,
                                       args.gamma,
                                       kernel_size,
                                       stride,
                                       args.minimum_num_state_rtgs_to_consider_for_important_state,
                                       args.window_size_for_important_state,
                                       args.notify_important_states_with_multiple_occurrences_n,
                                       args.first_visit_state_action,
                                       args.life_loss_negative_reward,
                                       args.steps_ahead_for_terminal_or_life_loss_state)

    if args.export_important_states_as_images:
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        loaded_important_states_indices = np.load(f"{file_name}.npz", allow_pickle=True)
        important_states_indices = loaded_important_states_indices['ims_indices']
        important_states_actions_rtg_differences = loaded_important_states_indices['ims_actions_rtg_differences']
        important_states = loaded_important_states_indices['states']

        idx_to_state = {i: important_states[i] for i in range(len(important_states))}

        print("Export important states to images has started")
        for i, ims_idx in enumerate(important_states_indices):
            cv2.imwrite(f"{file_name}/{important_states_actions_rtg_differences[i][1]}_rtg_diff_actions_{important_states_actions_rtg_differences[i][0]}_transition_{ims_idx}_i_{i}.png",
                        cv2.resize(idx_to_state[ims_idx][3], (512, 512)))
        print("Export important states to images has finished")
    else:
        start_time = datetime.now()
        # dd/mm/YY H:M:S
        start_time = start_time.strftime("%d/%m/%Y %H:%M:%S")

        # generate important states
        generate_important_states(args)

        with open(f"{file_name}.txt", "a") as file:
            end_time = datetime.now()
            # dd/mm/YY H:M:S
            end_time = end_time.strftime("%d/%m/%Y %H:%M:%S")

            file_lines = [f"job-id: {args.job_id}\n", f"start time: {start_time}\n", f"end time: {end_time}\n"]
            file.writelines(file_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=30)
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers')
    parser.add_argument('--data_dir_prefix', type=str, default='../dqn_replay/')
    parser.add_argument('--augment_only_sparse_reward_with_synthetic_oracle_feedback', action="store_true",
                        default=False, help="Augment only sparse reward with synthetic oracle feedbacks")
    parser.add_argument('--generate_dataset_seed', type=int, default=123)
    parser.add_argument('--job_id', type=int, default=None)
    parser.add_argument('--export_important_states_as_images', action="store_true",
                        default=False, help="convert important states indices to images")
    parser.add_argument('--run_in_parallel', type=int, default=0, help="load trajectories from buffer in parallel")
    parser.add_argument('--minimum_num_state_rtgs_to_consider_for_important_state', type=int, default=0, help="for a state to be considered important, it must have a minimum number of return-to-go")
    parser.add_argument('--window_size_for_important_state', type=int, default=0, help="Window size for important states, any other important states within this window with a smaller RTG will be removed")
    parser.add_argument('--notify_important_states_with_multiple_occurrences_n', type=int, default=0, help="Only include the first occurrence of an important state or all of its n occurrences")
    parser.add_argument('--first_visit_state_action', action="store_true", default=True, help="should perform first visit state action or every visit MC")
    parser.add_argument('--gpu', action="store_true", default=False, help='Whether to utilize GPU to run on, default is false which means CPU only')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--kernel_size_and_stride', nargs='+', type=int, default=[7, 7])

    parser.add_argument('--life_loss_negative_reward', type=int, default=0)
    parser.add_argument('--steps_ahead_for_terminal_or_life_loss_state', type=int, default=0)

    _args = parser.parse_args()

    run(_args)
