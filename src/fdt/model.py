import numpy as np
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch

import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                             .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformerGPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self,
                 vocab_size,
                 max_timestep,
                 game,
                 model_type,
                 feedback_regularization_lambda: float = 0.,
                 ignore_negative_feedback_regularization: bool = False,
                 disable_dt_loss_for_feedback_fine_tuning: bool = False,
                 turn_based_dt_loss_for_feedback_fine_tuning: bool = False,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=128,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_layer=6,
                 n_head=8,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 seed=123):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size + 1, n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, max_timestep + 1, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.block_size = self.config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                           nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                           nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                           nn.Flatten(), nn.Linear(3136, n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(vocab_size, n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        self.automatic_optimization = False
        self.epoch = 0

    def get_block_size(self):
        return self.block_size

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        state_embeddings = self.state_encoder(
            states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous())  # (batch * block_size, n_embd)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],
                                                    self.config.n_embd)  # (batch, block_size, n_embd)

        if actions is not None and self.config.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)# .type_as(states)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        elif actions is None and self.config.model_type == 'reward_conditioned':  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1] * 2, self.config.n_embd),
                                           dtype=torch.float32, device=state_embeddings.device)#.type_as(states)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.config.model_type == 'naive':
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)#.type_as(states)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
        elif actions is None and self.config.model_type == 'naive':  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)  # batch_size, traj_length, n_embd

        #position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, : token_embeddings.shape[1], :] TODO

        position_embeddings = []
        sample_id = 0
        for time in timesteps.flatten():
            position_embeddings.append(all_global_pos_emb[sample_id][time.item()].unsqueeze(0).unsqueeze(0))
            sample_id += 1

        position_embeddings = torch.cat(position_embeddings, dim=0) + self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.config.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.config.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.config.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.config.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        return logits

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        self.train()
        with torch.set_grad_enabled(True):
            states, actions, rtgs, timesteps, stepwise_feedbacks, regularized_feedbacks, stepwise_ims_largest_action = batch
            targets = actions
            # same action as inference
            logits = self(states, actions, targets, rtgs, timesteps)

            # if we are given some desired targets also calculate the loss
            loss = 0.
            if targets is not None:
                feedback_num = torch.count_nonzero(torch.greater_equal(stepwise_feedbacks, 0))
                positive_feedback_num = torch.count_nonzero(torch.greater_equal(stepwise_feedbacks, 1))
                feedback_num_used_in_regularization = feedback_num.clone()
                stepwise_ims_largest_action_num = torch.count_nonzero(torch.greater_equal(stepwise_ims_largest_action, 0))

                if self.config.feedback_regularization_lambda and torch.is_nonzero(feedback_num):
                    mask = regularized_feedbacks.reshape(-1) > 0

                    log_softmax = nn.LogSoftmax(dim=2)
                    softmax = nn.Softmax(dim=2)
                    log_preds_batch = log_softmax(logits)
                    log_preds_batch = log_preds_batch.reshape(-1, log_preds_batch.size(-1))

                    preds_batch = softmax(logits)
                    preds_batch = preds_batch.reshape(-1, preds_batch.size(-1))

                    actions_reshape = actions.reshape(-1)
                    stepwise_feedbacks_reshape = stepwise_feedbacks.reshape(-1)
                    stepwise_ims_largest_action_reshape = stepwise_ims_largest_action.reshape(-1)

                    feedback_regularization_loss = 0

                    for i in range(len(actions_reshape)):
                        action = actions_reshape[i]
                        feedback = stepwise_feedbacks_reshape[i]

                        if feedback == -1:
                            continue

                        if feedback.item() > 0:
                            feedback_regularization_loss = feedback_regularization_loss + -log_preds_batch[i][action]  # positive feedback loss
                        else:
                            if self.config.ignore_negative_feedback_regularization:
                                mask[i] = True
                            else:
                                ims_largest_action = stepwise_ims_largest_action_reshape[i]
                                if ims_largest_action != -1 and not torch.equal(action, ims_largest_action):
                                    feedback_regularization_loss = feedback_regularization_loss + -log_preds_batch[i][ims_largest_action]
                                else:
                                    if torch.is_nonzero(stepwise_ims_largest_action_num):
                                        feedback_num_used_in_regularization -= 1
                                        mask[i] = True
                                    else: # random feedback selection mode
                                        feedback_regularization_loss = feedback_regularization_loss + -torch.log(1 - preds_batch[i][action])

                    filtered_logits = logits.reshape(-1, logits.size(-1))[mask]
                    filtered_targets = targets.reshape(-1)[mask]
                    loss = F.cross_entropy(filtered_logits, filtered_targets)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                if self.config.ignore_negative_feedback_regularization and torch.is_nonzero(positive_feedback_num):
                    loss = loss + self.config.feedback_regularization_lambda * (feedback_regularization_loss / positive_feedback_num)
                else:
                    if torch.is_nonzero(feedback_num_used_in_regularization):
                        loss = loss + self.config.feedback_regularization_lambda * (feedback_regularization_loss / feedback_num_used_in_regularization)

        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()

        self.log("train/loss_step", loss.item(), prog_bar=True, on_step=True)
        self.logger.experiment.log({"train/loss_step": loss.item()})

        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch += 1

        epoch_loss = np.array([x["loss"].item() for x in outputs]).mean()
        last_step_loss = outputs[-1]["loss"].item()

        print()
        print(f"Epoch: {self.epoch}")
        print(f"epoch loss: {epoch_loss}")
        print(f"last step loss: {last_step_loss}")

        # self.log("train/loss_epoch", epoch_loss, on_step=False, on_epoch=True) TODO
        # self.log("train/loss_last_step_epoch", last_step_loss, on_step=False, on_epoch=True)

        #self.logger.experiment.log({"train/loss_epoch": epoch_loss, "epoch": self.epoch})
        #self.logger.experiment.log({"train/loss_last_step_epoch": last_step_loss, "epoch": self.epoch})
