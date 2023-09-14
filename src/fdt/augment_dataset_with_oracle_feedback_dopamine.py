import argparse
import gzip
import os
import cv2

import gin
import numpy as np
import tensorflow as tf
import tf_slim
from absl import logging

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import run_experiment
from fixed_replay_buffer import FixedReplayBuffer, STORE_FILENAME_PREFIX


class OracleDQNAgent(dqn_agent.DQNAgent):
    """Wrapper dqn agent to select actions for oracle feedbacks"""

    def __init__(self, sess, num_actions, summary_writer=None):
        super(OracleDQNAgent, self).__init__(sess, num_actions,
                                             summary_writer=summary_writer, epsilon_eval=0, eval_mode=True)

    def _select_action(self):
        action = super(OracleDQNAgent, self)._select_action()
        return action

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        if use_legacy_checkpoint:
            variables_to_restore = atari_lib.maybe_transform_variable_names(
                tf.compat.v1.global_variables(), legacy_checkpoint_load=True)
        else:
            global_vars = set([x.name for x in tf.compat.v1.global_variables()])
            ckpt_vars = [
                '{}:0'.format(name)
                for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            include_vars = list(global_vars.intersection(set(ckpt_vars)))
            variables_to_restore = tf_slim.get_variables_to_restore(
                include=include_vars)
        if variables_to_restore:
            reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
            reloader.restore(self._sess, checkpoint_path)
            logging.info('Done restoring from %s', checkpoint_path)
        else:
            logging.info('Nothing to restore!')

class OracleC51Agent(rainbow_agent.RainbowAgent):
    """Wrapper dqn agent to select actions for oracle feedbacks"""

    def __init__(self, sess, num_actions, summary_writer=None):
        super(OracleC51Agent, self).__init__(sess, num_actions,
                                             summary_writer=summary_writer, epsilon_eval=0, epsilon_train=0, replay_scheme='uniform', update_horizon=1, epsilon_fn=dqn_agent.identity_epsilon,)

    def _select_action(self):
        action = super(OracleC51Agent, self)._select_action()
        return action

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        if use_legacy_checkpoint:
            variables_to_restore = atari_lib.maybe_transform_variable_names(
                tf.compat.v1.global_variables(), legacy_checkpoint_load=True)
        else:
            global_vars = set([x.name for x in tf.compat.v1.global_variables()])
            ckpt_vars = [
                '{}:0'.format(name)
                for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            include_vars = list(global_vars.intersection(set(ckpt_vars)))
            variables_to_restore = tf_slim.get_variables_to_restore(
                include=include_vars)
        if variables_to_restore:
            reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
            reloader.restore(self._sess, checkpoint_path)
            logging.info('Done restoring from %s', checkpoint_path)
        else:
            logging.info('Nothing to restore!')


class FeedbackRunner(run_experiment.Runner):
    """Runner class to generate feedbacks"""

    def __init__(self, base_dir, trained_agent_ckpt_path, create_agent_fn,
                 use_legacy_checkpoint=False):
        self._trained_agent_ckpt_path = trained_agent_ckpt_path
        self._use_legacy_checkpoint = use_legacy_checkpoint
        super(FeedbackRunner, self).__init__(base_dir, create_agent_fn)

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        self._agent.reload_checkpoint(self._trained_agent_ckpt_path,
                                      self._use_legacy_checkpoint)
        self._start_iteration = 0


def create_dqn_agent(sess, environment, summary_writer=None):
    return OracleDQNAgent(sess, num_actions=environment.action_space.n,
                          summary_writer=summary_writer)

def create_c51_agent(sess, environment, summary_writer=None):
    return OracleC51Agent(sess, num_actions=environment.action_space.n,
                          summary_writer=summary_writer)

def create_runner(base_dir,
                  trained_agent_ckpt_path,
                  agent='dqn',
                  use_legacy_checkpoint=False):
    create_agent = create_dqn_agent if agent == 'dqn' else create_c51_agent
    return FeedbackRunner(base_dir, trained_agent_ckpt_path, create_agent,
                          use_legacy_checkpoint)


class FeedbackOracleModel:
    def __init__(self,
                 feedback_runner: FeedbackRunner):

        self.runner = feedback_runner
        self.model = self.runner._agent

    @staticmethod
    def save_feedback(checkpoint_dir, buffer_num, feedbacks):
        if not tf.io.gfile.exists(checkpoint_dir):
            return

        attr = STORE_FILENAME_PREFIX + "feedback"
        filename = os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(attr, buffer_num))

        with tf.io.gfile.GFile(filename, 'wb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                if attr.startswith(STORE_FILENAME_PREFIX):
                    np.save(outfile, feedbacks, allow_pickle=False)

    def select_action(self, state):
        # set oracle model's state
        self.model.state = state
        # select action based on given state
        oracle_action = self.model._select_action()

        return oracle_action

    def collect_and_save_oracle_feedbacks(self, buffer_num, game, data_dir_prefix, dataset_size: int = 1000000):

        print(f"collect oracle's feedbacks for dataset process has started with buffer_num={buffer_num}, game={game}, data_dir_prefix={data_dir_prefix}, dataset_size={dataset_size}")

        data_dir = os.sep.join([data_dir_prefix, game, '1', 'replay_logs'])
        frb = FixedReplayBuffer(
            data_dir=data_dir,
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=1000000)

        feedbacks = []
        oracle_actions = []
        trajectory_number = 0

        dataset_size = dataset_size if frb._replay_buffers[0].add_count > dataset_size else frb._replay_buffers[0].add_count.item()
        for i in range(dataset_size):
            if not frb._replay_buffers[0].is_valid_transition(i):
                feedbacks += [0]
                continue

            states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])

            # set oracle model's state
            self.model.state = states
            # select action based on given state
            oracle_action = self.model._select_action()

            # feedback logic, if oracle selected action is equal to the chosen action of the current transition then feedback is 1
            if oracle_action == ac[0]:
                feedbacks += [1]
            else:
                feedbacks += [0]

            # store selected action
            oracle_actions.append(oracle_action)

            if i % 100000 == 0:
                print(i)
                img = states.transpose((0, 3, 1, 2))[0][3]
                img = cv2.resize(img, (254, 254))
                cv2.imwrite(f"buffer_num_{buffer_num}_trajectory_{trajectory_number}_transition_{i}_action_{ac[0]}_next_action_{next_action[0]}_oracle_action_{oracle_action}_feedback_{feedbacks[-1]}.png", img)

            if terminal[0]:
                trajectory_number += 1

        # print statistics
        unique_oracle_actions, counts_oracle_actions = np.unique(np.array(oracle_actions), return_counts=True)
        unique_feedbacks, counts_feedbacks = np.unique(np.array(feedbacks), return_counts=True)
        print(f"buffer_num={buffer_num}, num of trajectories is: {trajectory_number}")
        print(f"buffer_num={buffer_num}, buffer size={frb._replay_buffers[0].add_count.item()}, dataset size={dataset_size}, feedbacks size={len(feedbacks)}")
        print(f"buffer_num={buffer_num}, oracle actions dist:")
        print(np.asarray((unique_oracle_actions, counts_oracle_actions)).T)
        print(f"buffer_num={buffer_num}, feedbacks dist:")
        print(np.asarray((unique_feedbacks, counts_feedbacks)).T)

        # save feedbacks to file
        FeedbackOracleModel.save_feedback(data_dir, buffer_num, np.array(feedbacks))


def run(agent,
        game,
        root_dir,
        restore_ckpt,
        use_legacy_checkpoint=False):
    """Main entrypoint for running and generating feedbacks.

    Args:
      agent: str, agent type to use.
      game: str, Atari 2600 game to run.
      root_dir: str, root directory where files will be stored.
      restore_ckpt: str, path to the checkpoint to reload.
      use_legacy_checkpoint: bool, whether to restore from a legacy (pre-Keras) checkpoint.
    """
    tf.compat.v1.reset_default_graph()

    config = """
    atari_lib.create_atari_environment.game_name = '{}'
    WrappedReplayBuffer.replay_capacity = 300
    """.format(game)

    base_dir = os.path.join(root_dir, 'agent_feedbacks', game, agent)
    gin.parse_config(config)

    # create feedback runner
    feedback_runner = create_runner(base_dir, restore_ckpt, agent, use_legacy_checkpoint)

    # initialize oracle feedback model
    feedback_oracle_model = FeedbackOracleModel(feedback_runner=feedback_runner)

    # collect feedbacks for all buffers
    for buffer_num in range(0, 51):
        feedback_oracle_model.collect_and_save_oracle_feedbacks(buffer_num, game, "./dqn_replay")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Breakout',
                        help="Name of the game environment to use")
    parser.add_argument('--agent_name', type=str, default='c51',
                        help="Name of pretrained agent to use")

    args = parser.parse_args()

    game = args.game.strip("'")
    agent_name = args.agent_name.strip("'")

    restore_ckpt = f'./synthetic_oracle/checkpoints/{agent_name}/{game}/1/tf_checkpoints/tf_ckpt-199'

    run(agent=agent_name,
        game=game,
        root_dir='dqn_feedback',
        restore_ckpt=restore_ckpt,
        use_legacy_checkpoint=True)
