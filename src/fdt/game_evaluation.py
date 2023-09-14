import concurrent
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from pytorch_lightning import seed_everything
from torch.nn import functional as F

from env import Env
from model import DecisionTransformerGPT
from utils import AgentPolicy, top_k_logits


class AtariGameEvaluationCallback(pl.Callback):

    def __init__(self,
                 run_get_returns_max_workers: int):
        super().__init__()

        self.run_get_returns_max_workers = run_get_returns_max_workers

        self.evaluation_return_metrics_per_epoch = []
        self.normalized_evaluation_return_metrics_per_epoch = []
        self.evaluation_return_metrics_wrapper = []
        self.normalized_evaluation_return_metrics_wrapper = []

    def generate_normalized_evaluation_return_metrics_per_epoch(self, game: str):
        return GameEvaluation.generate_normalized_evaluation_returns_metrics(game, self.evaluation_return_metrics_per_epoch)

    def on_fit_end(self, trainer, pl_module):
        game = pl_module.config.game

        self.normalized_evaluation_return_metrics_per_epoch = self.generate_normalized_evaluation_return_metrics_per_epoch(game)

        self.evaluation_return_metrics_wrapper = generate_metrics_wrapper(self.evaluation_return_metrics_per_epoch)
        self.normalized_evaluation_return_metrics_wrapper = generate_metrics_wrapper(self.normalized_evaluation_return_metrics_per_epoch)

        print()
        print("Training statistics:", flush=True)

        for evaluation_return_metric in self.evaluation_return_metrics_wrapper:
            print_and_log_statistics(evaluation_return_metric, pl_module.logger)
        for evaluation_return_metric in self.normalized_evaluation_return_metrics_wrapper:
            print_and_log_statistics(evaluation_return_metric, pl_module.logger, "normalized_")

    def on_train_epoch_end(self, trainer, pl_module):
        game_evaluation = GameEvaluation(self.run_get_returns_max_workers)
        game = pl_module.config.game

        if pl_module.config.model_type == 'naive':
            eval_return, _ = game_evaluation.get_returns(pl_module, game)
        elif pl_module.config.model_type == 'reward_conditioned':
            evaluation_return_metrics = game_evaluation.get_returns(pl_module, game)
        else:
            raise NotImplementedError()

        # store eval returns scores
        self.evaluation_return_metrics_per_epoch.append(evaluation_return_metrics)


@dataclass
class EvaluationReturnMetricWrapper:
    metric: str
    seed: int
    eval_return_per_epoch: np.array
    eval_return_per_episode: np.array
    agent_policy: AgentPolicy
    no_op_max: int
    repeat_action_probability: float


@dataclass
class EvaluationReturnMetric:
    metric: str
    seed: int
    eval_return: float
    eval_return_per_episode: np.array
    agent_policy: AgentPolicy
    no_op_max: int
    repeat_action_probability: float


def generate_metrics_wrapper(evaluation_return_metrics_per_epoch):
    evaluation_return_metrics_wrapper = []

    for metric_index in range(len(evaluation_return_metrics_per_epoch[0])):
        metric_name = evaluation_return_metrics_per_epoch[0][metric_index].metric
        seed = evaluation_return_metrics_per_epoch[0][metric_index].seed
        agent_policy = evaluation_return_metrics_per_epoch[0][metric_index].agent_policy
        no_op_max = evaluation_return_metrics_per_epoch[0][metric_index].no_op_max
        repeat_action_probability = evaluation_return_metrics_per_epoch[0][metric_index].repeat_action_probability

        eval_return_per_epoch = []
        eval_return_per_episode = []

        for epoch_index in range(len(evaluation_return_metrics_per_epoch)):
            eval_return_per_epoch.append(evaluation_return_metrics_per_epoch[epoch_index][metric_index].eval_return)
            eval_return_per_episode.append(evaluation_return_metrics_per_epoch[epoch_index][metric_index].eval_return_per_episode)

        metric_wrapper = EvaluationReturnMetricWrapper(metric_name,
                                                       seed,
                                                       np.array(eval_return_per_epoch),
                                                       np.concatenate(eval_return_per_episode),
                                                       agent_policy, no_op_max,
                                                       repeat_action_probability)
        evaluation_return_metrics_wrapper.append(metric_wrapper)

    return evaluation_return_metrics_wrapper


def print_and_log_statistics(evaluation_return_metric_: EvaluationReturnMetricWrapper, logger_, prefix: str = "",
                             per_suffix: str = "epoch", should_print: bool = True):
    metric = evaluation_return_metric_.metric
    seed_prefix = f"seed={evaluation_return_metric_.seed}_" if per_suffix == "epoch" else ""

    if should_print:
        print()
        print(f"{seed_prefix}{prefix}{metric}:")
        print(f"evaluation return per {per_suffix}: {evaluation_return_metric_.eval_return_per_epoch}", flush=True)
        print(f"best evaluation return: {np.amax(evaluation_return_metric_.eval_return_per_epoch)}", flush=True)
        print(f"evaluation return avg: {evaluation_return_metric_.eval_return_per_epoch.mean()}, std: {evaluation_return_metric_.eval_return_per_epoch.std()}", flush=True)
        print(f"evaluation return per episode avg: {evaluation_return_metric_.eval_return_per_episode.mean()}, std: {evaluation_return_metric_.eval_return_per_episode.std()}", flush=True)

    if logger_:
        logger_.experiment.summary[f"{seed_prefix}{prefix}best_{metric}_evaluation_return"] = np.amax(evaluation_return_metric_.eval_return_per_epoch)
        logger_.experiment.summary[f"{seed_prefix}{prefix}{metric}_evaluation_return_per_{per_suffix}"] = np.array2string(evaluation_return_metric_.eval_return_per_epoch)
        logger_.experiment.summary[f"{seed_prefix}{prefix}{metric}_evaluation_return_avg"] = evaluation_return_metric_.eval_return_per_epoch.mean()
        logger_.experiment.summary[f"{seed_prefix}{prefix}{metric}_evaluation_return_std"] = evaluation_return_metric_.eval_return_per_epoch.std()
        logger_.experiment.summary[f"{seed_prefix}{prefix}{metric}_evaluation_return_per_episode_avg"] = evaluation_return_metric_.eval_return_per_episode.mean()
        logger_.experiment.summary[f"{seed_prefix}{prefix}{metric}_evaluation_return_per_episode_std"] = evaluation_return_metric_.eval_return_per_episode.std()


class GameEvaluation:
    def __init__(self,
                 run_get_returns_max_workers: int,
                 rtg: int = None):
        self.run_get_returns_max_workers = run_get_returns_max_workers
        self.rtg = rtg

    @staticmethod
    def normalize_score(game: str, score: float) -> float:  # TODO
        if game == 'Breakout':
            random = 2
            gamer = 30
        elif game == 'Seaquest':
            random = 68
            gamer = 42055
        elif game == 'Qbert':
            random = 164
            gamer = 13455
        elif game == 'Pong':
            random = -21
            gamer = 15
        else:
            raise NotImplementedError()

        return 100 * ((score - random) / (gamer - random))

    @staticmethod
    def get_target_return(game, rtg):
        if game == 'Breakout':
            target_return = 90 if rtg is None else rtg
        elif game == 'Seaquest':
            target_return = 1150 if rtg is None else rtg
        elif game == 'Qbert':
            target_return = 14000 if rtg is None else rtg
        elif game == 'Pong':
            target_return = 20 if rtg is None else rtg
        else:
            raise NotImplementedError()

        return target_return

    @staticmethod
    def generate_normalized_evaluation_returns_metrics(game: str, evaluation_return_metrics_per_sequence: List[List[int]]) -> List[List[EvaluationReturnMetric]]:
        return [[EvaluationReturnMetric(evaluation_return_metric.metric, evaluation_return_metric.seed,
                                       GameEvaluation.normalize_score(game, evaluation_return_metric.eval_return),
                                       evaluation_return_metric.eval_return_per_episode,
                                       evaluation_return_metric.agent_policy,
                                       evaluation_return_metric.no_op_max,
                                       evaluation_return_metric.repeat_action_probability)
                 for evaluation_return_metric in evaluation_return_metrics]
                for evaluation_return_metrics in evaluation_return_metrics_per_sequence]

    def get_test_return(self, model, device, game, eval_num_iteration=10, seeds=None, logger=None):
        if seeds is None:
            seeds = [123, 231, 312]

        evaluation_return_metrics_per_seed = []

        model.eval().to(device)

        for seed in seeds:
            # release all the GPU memory cache that can be freed
            torch.cuda.empty_cache()

            print()
            print(f"Test statistics for seed: {seed}", flush=True)

            seed_everything(seed)

            try:
                evaluation_return_metrics = self.get_returns(model, game, num_iteration=eval_num_iteration, train_mode=False, seeds=[seed])
            except Exception:
                if self.run_get_returns_max_workers <= 1:
                    raise
                else:
                    torch.cuda.empty_cache()
                    # if failed to run in parallel, try without
                    self.run_get_returns_max_workers = 0
                    evaluation_return_metrics = self.get_returns(model, game, num_iteration=eval_num_iteration,
                                                                 train_mode=False, seeds=[seed])

            evaluation_return_metrics_per_seed.append(evaluation_return_metrics)

        normalized_evaluation_return_metrics_per_seed = GameEvaluation.generate_normalized_evaluation_returns_metrics(game, evaluation_return_metrics_per_seed)

        test_evaluation_return_metrics_wrapper = generate_metrics_wrapper(evaluation_return_metrics_per_seed)
        test_normalized_evaluation_return_metrics_wrapper = generate_metrics_wrapper(normalized_evaluation_return_metrics_per_seed)

        print()
        print(f"Test statistics average:", flush=True)
        for test_evaluation_return_metric_ in test_evaluation_return_metrics_wrapper:
            print_and_log_statistics(test_evaluation_return_metric_, logger, "test_", "seed")

        for test_evaluation_return_metric_ in test_normalized_evaluation_return_metrics_wrapper:
            print_and_log_statistics(test_evaluation_return_metric_, logger, "test_normalized_", "seed", should_print=False)

    def get_returns_per_seed(self,
                             task_index,
                             target_return,
                             device,
                             game,
                             seed,
                             num_iteration,
                             agent_policy,
                             no_op_max,
                             repeat_action_probability,
                             model_config,
                             state_dict_model_name):
        model = DecisionTransformerGPT(model_config.vocab_size,
                                       model_config.max_timestep,
                                       model_config.game,
                                       model_config.model_type,
                                       feedback_auxiliary_loss_lambda=model_config.feedback_auxiliary_loss_lambda,
                                       feedback_regularization_lambda=model_config.feedback_regularization_lambda,
                                       learning_rate=model_config.learning_rate,
                                       block_size=model_config.block_size,
                                       n_layer=model_config.n_layer,
                                       n_head=model_config.n_head,
                                       n_embd=model_config.n_embd,
                                       seed=model_config.seed)

        model.load_state_dict(torch.load(state_dict_model_name))
        model.eval().to(device)

        rng = np.random.RandomState(seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_iteration_ = 1 if (agent_policy is AgentPolicy.DETERMINISTIC and no_op_max is None) else num_iteration

        eval_return, eval_return_per_episode = self._get_returns(target_return,
                                                                 model,
                                                                 generator,
                                                                 rng,
                                                                 game=game,
                                                                 seed=seed,
                                                                 num_iteration=num_iteration_,
                                                                 agent_policy=agent_policy,
                                                                 no_op_max=no_op_max,
                                                                 repeat_action_probability=repeat_action_probability)

        metric = f"policy={agent_policy.name}_no-op={no_op_max}"

        return (metric, eval_return, eval_return_per_episode), task_index

    def get_returns_parallel(self, model, game, num_iteration: int = 10, seeds: List[int] = None,
                             train_mode: bool = True, max_workers: int = 6):
        if seeds is None:
            seeds = [123, 231, 312]

        target_return = GameEvaluation.get_target_return(game, self.rtg)

        evaluation_return_metrics = []

        if train_mode:
            model.eval()
            torch.set_grad_enabled(False)

        model_id = str(uuid.uuid4())
        state_dict_model_name = f"state_dict_model_{model_id}.pt"
        torch.save(model.state_dict(), state_dict_model_name)

        print()
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
            for seed in seeds:
                tasks = []
                task_index = 0

                for agent_policy in AgentPolicy:
                    for no_op_max in [30, None]:
                        for repeat_action_probability in [0]:
                            task_index += 1
                            tasks.append((task_index, target_return, model.device, game, seed, num_iteration, agent_policy, no_op_max, repeat_action_probability, model.config, state_dict_model_name))

                futures = {executor.submit(self.get_returns_per_seed, *task): task for task in tasks}

                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result, index = future.result()
                        results.append([result, index])
                    except Exception as e:
                        print(f"get_returns computation has failed with exception: {e}", flush=True)
                        raise

                results = sorted(results, key=lambda x: x[-1])

                for result in results:
                    metric, eval_return, eval_return_per_episode = result[0]
                    evaluation_return_metrics.append(EvaluationReturnMetric(metric,
                                                                            seed,
                                                                            eval_return,
                                                                            np.array(eval_return_per_episode),
                                                                            agent_policy,
                                                                            no_op_max,
                                                                            repeat_action_probability))

        for evaluation_return_metric in evaluation_return_metrics:
            print(f"seed: {evaluation_return_metric.seed}", flush=True)
            print(f"target return: {target_return}, evaluation return: {evaluation_return_metric.eval_return}, evaluation return per episode: {evaluation_return_metric.eval_return_per_episode}, metric: {evaluation_return_metric.metric}", flush=True)

        if train_mode:
            torch.set_grad_enabled(True)
            model.train()

        if os.path.exists(state_dict_model_name):
            os.remove(state_dict_model_name)

        return evaluation_return_metrics

    def get_returns(self, model, game, num_iteration: int = 10, seeds: List[int] = None, train_mode: bool = True):
        if self.run_get_returns_max_workers > 0:
            return self.get_returns_parallel(model, game, num_iteration, seeds, train_mode, max_workers=self.run_get_returns_max_workers)

        if seeds is None:
            seeds = [123, 231, 312]

        if model.config.model_type != 'naive':
            target_return = GameEvaluation.get_target_return(game, self.rtg)
        else:
            print(f"running model type naive")
            target_return = 0

        evaluation_return_metrics = []

        # disable grads + batchnorm + dropout
        if train_mode:
            model.eval()
            torch.set_grad_enabled(False)

        print()
        for seed in seeds:
            print(f"seed: {seed}", flush=True)
            for agent_policy in AgentPolicy:
                for no_op_max in [30, None]:
                    for repeat_action_probability in [0]:
                        rng = np.random.RandomState(seed)
                        generator = torch.Generator(device=model.device)
                        generator.manual_seed(seed)

                        num_iteration_ = 1 if (agent_policy is AgentPolicy.DETERMINISTIC and no_op_max is None) else num_iteration

                        eval_return, eval_return_per_episode = self._get_returns(target_return, model, generator, rng,
                                                                                 game=game,
                                                                                 seed=seed,
                                                                                 num_iteration=num_iteration_,
                                                                                 agent_policy=agent_policy,
                                                                                 no_op_max=no_op_max,
                                                                                 repeat_action_probability=repeat_action_probability)

                        metric = f"policy={agent_policy.name}_no-op={no_op_max}"
                        print(f"target return: {target_return}, evaluation return: {eval_return}, evaluation return per episode: {eval_return_per_episode}, metric: {metric}", flush=True)

                        evaluation_return_metrics.append(EvaluationReturnMetric(metric,
                                                                                seed,
                                                                                eval_return,
                                                                                np.array(eval_return_per_episode),
                                                                                agent_policy,
                                                                                no_op_max,
                                                                                repeat_action_probability))

        if train_mode:
            torch.set_grad_enabled(True)
            model.train()

        return evaluation_return_metrics

    def _choose_action(self, model, x, steps, generator: torch.Generator, rng: np.random.RandomState, temperature=1.0,
                       agent_policy=AgentPolicy.SAMPLE_FROM_DISTRIBUTION, top_k=None,
                       actions=None, rtgs=None, timesteps=None):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        block_size = model.get_block_size()

        for k in range(steps):
            # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed TODO
            x_cond = x if x.size(1) <= block_size // 3 else x[:, -block_size // 3:]  # crop context if needed
            if actions is not None:
                actions = actions if actions.size(1) <= block_size // 3 else actions[:,
                                                                             -block_size // 3:]  # crop context if needed
            rtgs = rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:,
                                                                -block_size // 3:]  # crop context if needed
            logits = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if agent_policy is AgentPolicy.SAMPLE_FROM_DISTRIBUTION:
                #ix = torch.multinomial(probs, num_samples=1, generator=generator)
                probs = probs.cpu().numpy()  # To support evaluation on CPU devices: changed sampling from torch.multinomial to numpy choice
                ix = torch.tensor([[rng.choice(len(probs[0]), size=1, p=probs[0])[0]]], device=model.device)
            elif agent_policy is AgentPolicy.EPSILON_GREEDY:
                sample = rng.random()
                eps_threshold = 0.001
                if sample > eps_threshold:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                else:
                    ix = torch.tensor([[rng.randint(model.config.vocab_size)]], device=model.device)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            x = ix

        return x

    def _get_returns(self, ret, model, generator: torch.Generator, rng: np.random.RandomState,
                     num_iteration=10, agent_policy=AgentPolicy.SAMPLE_FROM_DISTRIBUTION,
                     game=None, seed=None, max_steps_per_episode=27000, no_op_max=30, repeat_action_probability=0):

        if game is None:
            game = model.config.game.lower()
        if seed is None:
            seed = model.config.seed

        if game.lower() == 'seaquest':
            max_steps_per_episode = 5000  # deterministic policy can reach the configured max step, so to make the run finish reduced the max steps for seaquest

        args = Args(game, model.device, seed, rng=rng, no_op_max=no_op_max, repeat_action_probability=repeat_action_probability)
        env = Env(args)
        env.eval()

        T_rewards = []
        done = True

        episodes_length = []
        ones_tensor = torch.ones((1, 1, 1), dtype=torch.int64, device=model.device)
        zeros_tensor = torch.zeros((1, 1, 1), dtype=torch.int64, device=model.device)

        with torch.no_grad():
            for i in range(num_iteration):
                state = env.reset()
                state_tensor = (torch.tensor(state, dtype=torch.float32, device=model.device).div_(255)).unsqueeze(0).unsqueeze(0)
                rtgs = [ret]
                # first state is from env, first rtg is target return, and first timestep is 0
                sampled_action = self._choose_action(model, state_tensor, 1, generator, rng,
                                                     temperature=1.0,
                                                     agent_policy=agent_policy,
                                                     actions=None,
                                                     rtgs=torch.tensor(rtgs, dtype=torch.long, device=model.device).unsqueeze(0).unsqueeze(-1),
                                                     timesteps=zeros_tensor)

                j = 0
                all_states = state_tensor

                all_rtg = torch.tensor(rtgs, dtype=torch.long, device=model.device).unsqueeze(0).unsqueeze(-1)
                all_actions = None

                actions = []

                while True:
                    if done:
                        state, reward_sum, done = env.reset(), 0, False

                    all_actions = sampled_action.unsqueeze(0) if all_actions is None else torch.cat([all_actions, sampled_action.unsqueeze(0)], dim=1)
                    action = sampled_action.cpu().numpy()[0, -1]
                    actions += [sampled_action]

                    # check for life loss and trigger fire if required
                    env.check_for_life_loss_and_initialize()

                    state, reward, done = env.step(action)

                    reward_sum += reward
                    j += 1

                    if done or j == max_steps_per_episode:
                        if j == max_steps_per_episode:
                            done = True
                            print(f"Reach max steps for policy: {agent_policy.name}, seed: {seed}", flush=True)

                        T_rewards.append(reward_sum)
                        break

                    state_tensor = (torch.tensor(state, dtype=torch.float32, device=model.device).div_(255)).unsqueeze(0).unsqueeze(0)

                    all_states = torch.cat([all_states, state_tensor], dim=0)

                    rtg_value = rtgs[-1] - reward
                    rtg_tensor = torch.tensor([rtg_value], dtype=torch.long, device=model.device).unsqueeze(0).unsqueeze(-1)
                    all_rtg = torch.cat([all_rtg, rtg_tensor], dim=1)

                    rtgs += [rtg_value]
                    # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in _choose_action method)
                    # timestep is just current timestep
                    sampled_action = self._choose_action(model, all_states.unsqueeze(0), 1, generator,
                                                         rng,
                                                         temperature=1.0,
                                                         agent_policy=agent_policy,
                                                         actions=all_actions,
                                                         rtgs=all_rtg,
                                                         timesteps=(min(j, model.config.max_timestep) * ones_tensor))

                    if game.lower() == 'pong' and reward != 0 and env.fire_action is not None and sampled_action.cpu().numpy()[0, -1] != env.fire_action:
                        sampled_action = torch.tensor([[env.fire_action]], device=model.device)

                episodes_length.append(j)
                print(f"iteration: {i}, total episode length: {episodes_length[i]}, policy: {agent_policy.name}", flush=True)  # TODO

        env.close()
        eval_return = int(sum(T_rewards) / num_iteration)

        return eval_return, T_rewards


class Args:
    def __init__(self,
                 game: str,
                 device: str,
                 seed: str,
                 rng: np.random.RandomState,
                 no_op_max: int = 30,
                 repeat_action_probability: float = 0):
        self.seed = seed
        self.device = device
        self.rng = rng
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
        self.no_op_max = no_op_max
        self.repeat_action_probability = repeat_action_probability
