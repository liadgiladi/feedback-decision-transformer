from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from create_dataset import DatasetStatistics, DQNDatasetCreator


class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, stepwise_feedbacks, stepwise_ims_largest_smallest_actions_rtg_differences):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.stepwise_feedbacks = stepwise_feedbacks
        self.stepwise_ims_largest_action = [actions[0] if actions is not None else -1 for actions in stepwise_ims_largest_smallest_actions_rtg_differences]
        self.stepwise_ims_smallest_action = [actions[1] if actions is not None else -1 for actions in stepwise_ims_largest_smallest_actions_rtg_differences]

        self.regularized_feedbacks = np.where(stepwise_feedbacks == 0, 0, 1)  # stepwise_feedbacks contains -1/0/1

    def __len__(self):
        return len(self.data) - self.block_size  # TODO consider to change, can be block_size // 3

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size  # protection against negative idx is placed in create dataset logic
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)  # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)
        stepwise_feedbacks = torch.tensor(self.stepwise_feedbacks[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        regularized_feedbacks = torch.tensor(self.regularized_feedbacks[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        stepwise_ims_largest_action = torch.tensor(self.stepwise_ims_largest_action[idx:done_idx], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps, stepwise_feedbacks, regularized_feedbacks, stepwise_ims_largest_action


def generate_dataset(args) -> Tuple[StateActionReturnDataset, dict, DatasetStatistics]:
    if not args.load_dataset_from_file:
        dqn_dataset_creator: DQNDatasetCreator = DQNDatasetCreator(args.num_buffers,
                                                                   args.num_steps,
                                                                   args.game,
                                                                   args.data_dir_prefix,
                                                                   args.trajectories_per_buffer,
                                                                   args.context_length,
                                                                   args.augment_reward_with_synthetic_oracle_feedback_prob,
                                                                   args.augment_only_sparse_reward_with_synthetic_oracle_feedback,
                                                                   args.subset_augment_reward_with_synthetic_oracle_feedback_prob,
                                                                   args.augment_reward_with_synthetic_oracle_feedback_rng_num,
                                                                   args.subset_augment_reward_with_synthetic_oracle_feedback_rng_num,
                                                                   args.states_for_feedbacks_based_on_important_states_filename,
                                                                   args.num_of_important_states,
                                                                   args.generate_dataset_seed)
        obss, actions, returns, done_idxs, rtgs, timesteps, feedbacks, state_action_positive_feedback_cache,\
        stepwise_feedbacks, is_step_allowed_for_feedback, stepwise_ims_largest_smallest_actions_rtg_differences, dataset_statistics = dqn_dataset_creator.create_dataset()

        np.savez_compressed(args.dataset_file_name.split(".")[0], obss=np.array(obss), actions=actions, returns=returns,
                            done_idxs=done_idxs, rtgs=rtgs, timesteps=timesteps, feedbacks=feedbacks,
                            state_action_positive_feedback_cache=np.array([state_action_positive_feedback_cache]),
                            stepwise_feedbacks=stepwise_feedbacks,
                            is_step_allowed_for_feedback=is_step_allowed_for_feedback,
                            stepwise_ims_largest_smallest_actions_rtg_differences=stepwise_ims_largest_smallest_actions_rtg_differences,
                            dataset_statistics=np.array([dataset_statistics]))
    else:
        loaded_dataset = np.load(args.dataset_file_name, allow_pickle=True)

        obss = loaded_dataset['obss']
        actions = loaded_dataset['actions']
        returns = loaded_dataset['returns']
        done_idxs = loaded_dataset['done_idxs']
        rtgs = loaded_dataset['rtgs']
        timesteps = loaded_dataset['timesteps']
        feedbacks = loaded_dataset['feedbacks']
        state_action_positive_feedback_cache = loaded_dataset['state_action_positive_feedback_cache'][0]
        stepwise_feedbacks = loaded_dataset['stepwise_feedbacks']
        is_step_allowed_for_feedback = loaded_dataset['is_step_allowed_for_feedback']
        stepwise_ims_largest_smallest_actions_rtg_differences = loaded_dataset['stepwise_ims_largest_smallest_actions_rtg_differences']
        dataset_statistics = loaded_dataset['dataset_statistics'][0]

    return StateActionReturnDataset(obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps, stepwise_feedbacks, stepwise_ims_largest_smallest_actions_rtg_differences),\
           state_action_positive_feedback_cache, dataset_statistics
