import argparse
import ast
import logging
import os
import uuid

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset import generate_dataset
from game_evaluation import AtariGameEvaluationCallback, GameEvaluation
from lr_decay import LearningRateDecayCallback
from model import DecisionTransformerGPT
from utils import set_seed


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"{}\" is not a list".format(s))
    return v


def fit_trainer(args, train_dataset, dataset_statistics, num_workers,
                deterministic, strategy, wandb_logger):
    def generate_callbacks():
        # callbacks
        checkpoint_dirpath = os.path.join(wandb_logger.experiment.dir, "checkpoints")
        checkpoint_filename = "fdt-" + args.game + "-{epoch:03d}"
        periodic_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dirpath,
            filename=checkpoint_filename,
            save_last=False,
            save_top_k=-1,
            every_n_epochs=1,
        )

        callbacks = [periodic_checkpoint_callback]

        progress_bar = TQDMProgressBar(refresh_rate=int((dataset_statistics.loaded_transitions / args.batch_size) - 1))

        callbacks.append(progress_bar)

        if not args.disable_lr_decay:
            lr_decay = LearningRateDecayCallback(learning_rate=args.lr, warmup_tokens=512 * 20,
                                                 final_tokens=2 * len(train_dataset) * args.context_length * 3)

            callbacks.append(lr_decay)

        if not args.disable_training_game_evaluation_callback:
            atari_game_eval_callback = AtariGameEvaluationCallback(args.run_get_returns_max_workers)
            callbacks.append(atari_game_eval_callback)

        return callbacks

    # generate data loader
    train_dataloader = generate_train_dataloader(train_dataset, args.batch_size, num_workers)

    # model
    model = generate_model(args, train_dataset)

    # callbacks
    callbacks = generate_callbacks()

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, callbacks=callbacks, deterministic=deterministic,
                         strategy=strategy, logger=wandb_logger)

    model_uuid = f"model-{uuid.uuid4()}.ckpt"
    try:
        trainer.fit(model, train_dataloader)
    except Exception as e:
        print("trainer fit operation has failed with exception: {}".format(e))
        raise

    # trainer.save_checkpoint(model_uuid)

    del train_dataset
    del train_dataloader

    return model, model_uuid


def generate_model(args, train_dataset):
    model = DecisionTransformerGPT(train_dataset.vocab_size, max(train_dataset.timesteps),
                                   args.game, args.model_type,
                                   args.feedback_regularization_lambda,
                                   args.ignore_negative_feedback_regularization,
                                   args.disable_dt_loss_for_feedback_fine_tuning,
                                   args.turn_based_dt_loss_for_feedback_fine_tuning,
                                   learning_rate=args.lr, block_size=train_dataset.block_size,
                                   n_layer=6, n_head=8, n_embd=128, seed=args.seed)

    return model


def load_model(args, run):
    # model
    model = DecisionTransformerGPT(0, 0,
                                   args.game, args.model_type,
                                   args.feedback_regularization_lambda,
                                   args.ignore_negative_feedback_regularization,
                                   args.disable_dt_loss_for_feedback_fine_tuning,
                                   args.turn_based_dt_loss_for_feedback_fine_tuning,
                                   learning_rate=args.lr, block_size=args.context_length * 3,
                                   n_layer=6, n_head=8, n_embd=128, seed=args.seed)

    api = wandb.Api()
    runs = api.runs(args.wandb_project_name_for_loading_pretrained_model)
    model_id = ""
    for _run in runs:
        if _run.State == 'failed' or _run.State == 'running' or _run.State == 'crashed':
            continue

        if 'eval_model' not in _run.config or _run.config['eval_model'] == False:
            if _run.config['seed'] == args.seed and _run.config[
                'generate_dataset_seed'] == args.generate_dataset_seed and \
                    _run.config['lr'] == args.lr and _run.config['context_length'] == args.context_length and \
                    _run.config['disable_lr_decay'] == args.disable_lr_decay and \
                    _run.config['game'] == args.game and _run.config['num_steps'] == args.num_steps and \
                    _run.config['num_buffers'] == args.num_buffers and _run.config['batch_size'] == args.batch_size and \
                    _run.config['trajectories_per_buffer'] == args.trajectories_per_buffer and \
                    _run.config[
                        'augment_reward_with_synthetic_oracle_feedback_prob'] == args.augment_reward_with_synthetic_oracle_feedback_prob and \
                    _run.config[
                        'augment_reward_with_synthetic_oracle_feedback_rng_num'] == args.augment_reward_with_synthetic_oracle_feedback_rng_num and \
                    _run.config[
                        'augment_only_sparse_reward_with_synthetic_oracle_feedback'] == args.augment_only_sparse_reward_with_synthetic_oracle_feedback and \
                    _run.config[
                        'subset_augment_reward_with_synthetic_oracle_feedback_prob'] == args.subset_augment_reward_with_synthetic_oracle_feedback_prob and \
                    _run.config[
                        'subset_augment_reward_with_synthetic_oracle_feedback_rng_num'] == args.subset_augment_reward_with_synthetic_oracle_feedback_rng_num and \
                    _run.config['feedback_regularization_lambda'] == args.feedback_regularization_lambda and \
                    _run.config[
                        'ignore_negative_feedback_regularization'] == args.ignore_negative_feedback_regularization and \
                    (_run.config[
                         'states_for_feedbacks_based_on_important_states_filename'] == args.states_for_feedbacks_based_on_important_states_filename or (
                             _run.config[
                                 'states_for_feedbacks_based_on_important_states_filename'] is None and args.states_for_feedbacks_based_on_important_states_filename == '')) and \
                    _run.config['num_of_important_states'] == args.num_of_important_states:
                model_id = f"model-{_run.id}:v{args.epochs - 1}"
                break

    assert model_id != ""
    print(f"Loading pretrained model: {model_id}", flush=True)

    artifact = run.use_artifact(f'{args.wandb_project_name_for_loading_pretrained_model}/{model_id}',
                                type='model')
    artifact.download()
    model = model.load_from_checkpoint(f'./artifacts/{model_id}/model.ckpt',
                                       feedback_regularization_lambda=args.feedback_regularization_lambda,
                                       ignore_negative_feedback_regularization=args.ignore_negative_feedback_regularization,
                                       disable_dt_loss_for_feedback_fine_tuning=args.disable_dt_loss_for_feedback_fine_tuning,
                                       turn_based_dt_loss_for_feedback_fine_tuning=args.turn_based_dt_loss_for_feedback_fine_tuning,
                                       learning_rate=args.lr)

    print("model:")
    print(model.config)

    return model


def generate_train_dataloader(train_dataset, batch_size, num_workers):
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  pin_memory=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True)

    return train_dataloader


def validate_args(args):
    if not args.augment_reward_with_synthetic_oracle_feedback_prob and args.subset_augment_reward_with_synthetic_oracle_feedback_prob:
        raise ValueError(
            "subset augment reward with synthetic oracle feedback can't be set without augment_reward_with_synthetic_oracle_feedback_prob")

    if args.dynamic_max_rtg and args.conditioned_rtg != -1:
        raise ValueError("dynamic_max_rtg param can't be set with conditioned_rtg param")

    if args.conditioned_rtg != -1 and args.conditioned_rtg < 1:
        raise ValueError("invalid conditioned_rtg argument")

    if args.feedback_regularization_fine_tuning and not args.feedback_regularization_lambda:
        raise ValueError(
            "feedback_regularization_fine_tuning param can't be set without feedback_regularization_lambda")

    if args.disable_dt_loss_for_feedback_fine_tuning and not args.feedback_regularization_fine_tuning:
        raise ValueError("disable_dt_loss_for_feedback_fine_tuning param can't be set without fine-tuning enabled")

    if args.disable_dt_loss_for_feedback_fine_tuning and args.turn_based_dt_loss_for_feedback_fine_tuning:
        raise ValueError(
            "disable_dt_loss_for_feedback_fine_tuning and turn_based_dt_loss_for_feedback_fine_tuning can't be both activated")

    if args.turn_based_dt_loss_for_feedback_fine_tuning and not args.feedback_regularization_fine_tuning:
        raise ValueError("turn_based_dt_loss_for_feedback_fine_tuning param can't be set without fine-tuning enabled")

    if args.states_for_feedbacks_based_on_important_states_filename and args.num_of_important_states <= 0:
        raise ValueError(
            f"states_for_feedbacks_based_on_important_states_filename param can't be set with num_of_important_states={args.num_of_important_states}")

    if not args.states_for_feedbacks_based_on_important_states_filename and args.num_of_important_states > 0:
        raise ValueError(
            f"num_of_important_states param can't be set without states_for_feedbacks_based_on_important_states_filename")

    if args.augment_reward_with_synthetic_oracle_feedback_prob and (
            args.states_for_feedbacks_based_on_important_states_filename or args.num_of_important_states > 0):
        raise ValueError(
            f"augment_reward_with_synthetic_oracle_feedback_prob param can't be set with important states flags")

    if args.eval_model and args.wandb_project_name_for_loading_pretrained_model == '':
        raise ValueError("eval_model param can't be set without wandb_project_name_for_loading_pretrained_model")


def initialize_tags(args):
    tags = [f"seed={args.seed}"]
    tags.append(f"generate_dataset_seed={args.generate_dataset_seed}")
    tags.append(f"num_steps={args.num_steps}")
    tags.append(f"epochs={args.epochs}")

    if args.feedback_regularization_lambda:
        tags.append(f"auxiliary-regularization-lambda-{args.feedback_regularization_lambda}")

    if args.feedback_regularization_fine_tuning:
        tags.append("feedback-regularization-fine-tuning")

    if args.disable_dt_loss_for_feedback_fine_tuning:
        tags.append("disable_dt_loss_for_feedback_fine_tuning")

    if args.turn_based_dt_loss_for_feedback_fine_tuning:
        tags.append("turn_based_dt_loss_for_feedback_fine_tuning")

    if not args.augment_reward_with_synthetic_oracle_feedback_prob and not args.feedback_regularization_lambda:
        tags.append("dt-baseline")

    if args.states_for_feedbacks_based_on_important_states_filename and args.num_of_important_states > 0:
        tags.append(f"ims-{args.num_of_important_states}")

    if args.eval_model:
        tags.append("eval-model")

    return tags


def run(args):
    validate_args(args)

    # disable torch debug APIs
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # make deterministic
    set_seed(args.seed)
    seed_everything(args.seed)

    args.model_type = args.model_type.strip("'")
    args.game = args.game.strip("'")
    args.data_dir_prefix = args.data_dir_prefix.strip("'")
    args.states_for_feedbacks_based_on_important_states_filename = args.states_for_feedbacks_based_on_important_states_filename.strip("'")

    num_workers = 4
    if torch.cuda.is_available():
        num_workers = args.num_workers

    # flags
    deterministic = False
    strategy = None
    device = "cpu"
    if args.gpus > 0:
        deterministic = True
        strategy = "dp"
        device = "cuda:0"

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )

    tags = initialize_tags(args)
    # initialize logger
    run = wandb.init(project=f"{args.game}_{args.wandb_project_name}", config=args, tags=tags)
    wandb_logger = WandbLogger(log_model=True)

    print(args, flush=True)

    if args.eval_model or args.feedback_regularization_fine_tuning:
        model = load_model(args, run)

    # training phase
    model_ckp_file = None
    if not args.eval_model and args.epochs > 0:
        # data
        train_dataset, state_action_positive_feedback_cache, dataset_statistics = generate_dataset(args)
        max_rtgs = max(train_dataset.rtgs)

        # store dataset statistics
        wandb_logger.experiment.config.update(dataset_statistics.__dict__)
        print(f"Dataset statistics: {dataset_statistics}", flush=True)

        model, model_ckp_file = fit_trainer(args, train_dataset, dataset_statistics,
                                     num_workers, deterministic, strategy, wandb_logger)

    # eval phase
    if args.eval_model:
        rtg = None
        if args.dynamic_max_rtg:
            rtg = max_rtgs
        elif args.conditioned_rtg != -1:
            rtg = args.conditioned_rtg

        game_evaluation = GameEvaluation(args.run_get_returns_max_workers,
                                         rtg=rtg)
        game_evaluation.get_test_return(model, device, args.game, seeds=args.test_seeds, logger=wandb_logger)

    # log dataset artifact
    #dataset_artifact = wandb.Artifact('dataset', type='dataset')
    #dataset_artifact.add_file(args.dataset_file_name)
    #wandb.log_artifact(dataset_artifact)

    if model_ckp_file is not None and os.path.exists(model_ckp_file):
        os.remove(model_ckp_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for reproducibility")
    parser.add_argument('--lr', type=float, default=6e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument('--context_length', type=int, default=30,
                        help="Number of states to include in each context")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument('--model_type', type=str, default='reward_conditioned',
                        help="Type of model to train")
    parser.add_argument('--num_steps', type=int, default=500000,
                        help="Total number of environment steps to collect")
    parser.add_argument('--num_buffers', type=int, default=50,
                        help="Number of replay buffers to use for generating training dataset")
    parser.add_argument('--game', type=str, default='Breakout',
                        help="Name of the game environment to use")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for training")
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help="Number of trajectories to sample from each of the buffers")
    parser.add_argument('--data_dir_prefix', type=str, default='../dqn_replay/',
                        help="Prefix directory for replay buffer dataset")

    parser.add_argument('--augment_reward_with_synthetic_oracle_feedback_prob', type=float, default=0,
                        help="How often to augment rewards with feedback, probability")
    parser.add_argument('--augment_reward_with_synthetic_oracle_feedback_rng_num', type=int, default=123,
                        help="Random seed for reward augmentation with feedback")
    parser.add_argument('--augment_only_sparse_reward_with_synthetic_oracle_feedback', action="store_true", default=False,
                        help="Augment only sparse reward with synthetic oracle feedbacks")
    parser.add_argument('--subset_augment_reward_with_synthetic_oracle_feedback_prob', type=float, default=0,
                        help="Second condition prob for subset augmented rewards with feedback")
    parser.add_argument('--subset_augment_reward_with_synthetic_oracle_feedback_rng_num', type=int, default=0,
                        help="Random seed for subset reward augmentation with feedback")

    parser.add_argument('--feedback_regularization_lambda', type=float, default=0,
                        help="Lambda value for feedback regularization loss in training")
    parser.add_argument('--ignore_negative_feedback_regularization', action="store_true", default=False,
                        help="Whether to ignore negative feedback for regularization loss in training")
    parser.add_argument('--feedback_regularization_fine_tuning', action="store_true", default=False,
                        help="Whether to perform feedback regularization fine-tuning after training the baseline")
    parser.add_argument('--disable_dt_loss_for_feedback_fine_tuning', action="store_true", default=False,
                        help="Whether to disable Decision-Transformer cross-entropy for fine-tuning after training the baseline")
    parser.add_argument('--turn_based_dt_loss_for_feedback_fine_tuning', action="store_true", default=False,
                        help="Whether to perform Decision-Transformer cross-entropy turn based, i.e., one epoch on, one epoch off for fine-tuning after training the baseline")

    parser.add_argument('--load_dataset_from_file', action="store_true", default=False,
                        help="If set, the dataset will be loaded from an offline file instead of generated on the fly.")
    parser.add_argument('--dataset_file_name', type=str, default='dataset.npz',
                        help="The name of the file to load the dataset from when '--load_dataset_from_file' is set.")
    parser.add_argument('--gpus', type=int, default=0,
                        help='The number of GPUs to train on. By default, the value is 0, which means the training will run on the CPU only.')
    parser.add_argument('--test_seeds', type=arg_as_list, default="[123, 231, 312, 0, 42, 84, 64, 128, 256, 512]",
                        help="A list of seed values for testing. By default, ten different seed values will be used.")
    parser.add_argument('--disable_training_game_evaluation_callback', action="store_true", default=False,
                        help="If set, game evaluation will not be performed after each epoch during training.")
    parser.add_argument('--run_get_returns_max_workers', type=int, default=0,
                        help="The number of workers to allocate for the get returns method.")
    parser.add_argument('--generate_dataset_seed', type=int, default=123,
                        help="The seed value used to generate the dataset.")
    parser.add_argument('--dynamic_max_rtg', action="store_true", default=False,
                        help="If set, dynamic max rtg will be used as the initial conditioned RTG.")
    parser.add_argument('--wandb_project_name', type=str, default='test',
                        help="The name of the Weights & Biases project to log to. A prefix of the name of the game will be added to the project name.")
    parser.add_argument('--wandb_project_name_for_loading_pretrained_model', type=str, default='',
                        help="The name of the Weights & Biases project to load the pretrained model from.")
    parser.add_argument('--disable_lr_decay', action="store_true", default=False,
                        help="If set, learning rate decay will not be applied.")
    parser.add_argument('--conditioned_rtg', type=int, default=-1,
                        help="The conditioned RTG to use for training. If not specified, the default RTG will be used.")
    parser.add_argument('--job_id', type=int, default=None,
                        help="The job ID associated with the current training run.")

    parser.add_argument('--states_for_feedbacks_based_on_important_states_filename', type=str, default='',
                        help="The name of the file containing the indices of important states. Feedbacks will be based on these states.")
    parser.add_argument('--num_of_important_states', type=int, default=-1,
                        help="The number of important states to use. If not specified, important states is disabled and random selection will be applied.")
    parser.add_argument('--eval_model', action="store_true", default=False,
                        help="If set, game evaluation will be performed using a pretrained model without training from scratch.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="The number of workers to use for parallel processing. The default value is 4.")

    args = parser.parse_args()

    run(args)
