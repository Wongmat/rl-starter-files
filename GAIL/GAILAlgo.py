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

        logs = {
            "return_per_episode": 0,
            "reshaped_return_per_episode": 0,
            "num_frames_per_episode": 0,
            "num_frames": len(actions)
        }
        obs = self.preprocess_obss(obs)

        return {'obs': obs, 'actions': actions}, logs

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_disc_loss = 0
        update_actor_loss = 0

        # Initialize memory

        self.acmodel.recurrent = False
        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        embedding = self.acmodel.embed_obs(exps['obs'])
        batch_size = embedding.shape[0]
        for i in range(0, embedding.shape[0], batch_size):
            # Create a sub-batch of experience

            # sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                batch_emb = embedding[i:i + batch_size]
                batch_act = exps['actions'][i:i + batch_size]
                exp_acts = torch.Tensor([int(act) for act in batch_act])
                exp_input = torch.cat((batch_emb, exp_acts.unsqueeze(1)), 1)

                act_acts = torch.argmax(self.acmodel.actor(batch_emb), dim=1)
                print(act_acts)
                act_input = torch.cat((batch_emb, act_acts.unsqueeze(1)), 1)
                self.optim_discriminator.zero_grad()

                exp_d_output = self.acmodel.discriminator(exp_input)
                act_d_output = self.acmodel.discriminator(act_input)
                d_loss = F.binary_cross_entropy(
                    exp_d_output,
                    torch.full((batch_size, 1), 1).float())

                d_loss += F.binary_cross_entropy(
                    act_d_output,
                    torch.full((batch_size, 1), 0).float())

                d_loss.backward(retain_graph=True)
                self.optim_discriminator.step()

                self.optim_actor.zero_grad()

                a_loss = -self.acmodel.discriminator(act_input).mean()
                a_loss.backward()
                self.optim_actor.step()

            # Update batch values

            update_disc_loss += d_loss.item()
            update_actor_loss += a_loss.item()

        # Log some values

        logs = {
            "discriminator loss": update_disc_loss,
            "actor loss": update_actor_loss,
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
