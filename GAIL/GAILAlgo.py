import numpy
import torch
import os
import torch.nn.functional as F
import pickle
from torch_ac.algos.base import BaseAlgo

TRAJ_FOLDER = 'example_trajs/'


class GAILAlgo(BaseAlgo):
    """The GAIL algorithm."""
    def __init__(self,
                 envs,
                 model,
                 device=None,
                 num_frames_per_proc=None,
                 discount=0.99,
                 lr=0.01,
                 gae_lambda=0.95,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5,
                 recurrence=4,
                 rmsprop_alpha=0.99,
                 rmsprop_eps=1e-8,
                 preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, model, device, num_frames_per_proc, discount,
                         lr, gae_lambda, entropy_coef, value_loss_coef,
                         max_grad_norm, recurrence, preprocess_obss,
                         reshape_reward)

        self.optim_discriminator = torch.optim.Adam(
            self.acmodel.discriminator.parameters(), lr=lr)

        self.optim_actor = torch.optim.Adam(self.acmodel.actor.parameters(),
                                            lr=lr)

    def collect_experiences(self):
        path = TRAJ_FOLDER + '/exp_traj0.pkl'
        obs = []
        actions = []
        counter = 0

        while os.path.exists(path):
            exp_traj = pickle.load(open(path, 'rb'))

            for state, action in exp_traj:
                obs.append(state)
                actions.append(action)

            counter += 1
            path = TRAJ_FOLDER + f'/exp_traj{counter}.pkl'

        logs = None

        obs = self.preprocess_obss(obs)

        return {'obs': obs, 'actions': actions}, logs

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        self.acmodel.recurrent = False
        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        embedding = self.acmodel.embed_obs(exps['obs'])
        batch_size = 8
        for i in range(0, embedding.shape[0], batch_size):
            # Create a sub-batch of experience

            # sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                batch_emb = embedding[i:i + batch_size]
                batch_act = exps['actions'][i:i + batch_size]
                int_acts = torch.Tensor([int(act) for act in batch_act])
                d_input = torch.cat((batch_emb, int_acts.unsqueeze(1)), 1)
                d_output = self.acmodel.discriminator(d_input)
                d_loss = F.binary_cross_entropy(
                    d_output,
                    torch.full((batch_size, 1), 1).float())

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(
            p.grad.data.norm(2)**2 for p in self.acmodel.parameters())**0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
