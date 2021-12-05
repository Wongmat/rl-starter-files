#!/usr/bin/env python3

import argparse
import gym
import pickle
from gym_minigrid.window import Window


class DemosWindow(Window):
    def __init__(self, title, env, args):
        super().__init__(title)
        self.curr_a_seq = []
        self.curr_s_seq = []
        self.saved_seqs = []
        self.env = env
        self.args = args
        self.reg_key_handler(self.key_handler)

    def redraw(self, img):
        if not self.args.agent_view:
            img = self.env.render('rgb_array', tile_size=self.args.tile_size)
            self.env.window.close()

        self.show_img(img)

    def reset(self):
        if self.args.seed != -1:
            self.env.seed(self.args.seed)

        obs = self.env.reset()

        if hasattr(self.env, 'mission'):
            print('Mission: %s' % self.env.mission)
            self.set_caption(self.env.mission)

        self.curr_a_seq = []
        self.curr_s_seq = []
        self.curr_s_seq.append(obs)
        self.redraw(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.curr_a_seq.append(action)
        self.curr_s_seq.append(obs)

        if done:
            print('done!')
            self.saved_seqs.append(list(zip(self.curr_s_seq, self.curr_a_seq)))
            self.reset()
        else:
            self.redraw(obs)

    def key_handler(self, event):
        print('pressed', event.key)

        if event.key == 'escape':
            for ix, seq in enumerate(self.saved_seqs):
                pickle.dump(seq, open(f'exp_traj{ix}.pkl', 'wb'))

            self.close()
            return

        if event.key == 'backspace':
            self.reset()
            return

        if event.key == 'left':
            self.step(self.env.actions.left)
            return
        if event.key == 'right':
            self.step(self.env.actions.right)
            return
        if event.key == 'up':
            self.step(self.env.actions.forward)
            return

        # Spacebar
        if event.key == ' ':
            self.step(self.env.actions.toggle)
            return
        if event.key == 'p':
            self.step(self.env.actions.pickup)
            return
        if event.key == 'd':
            self.step(self.env.actions.drop)
            return

        if event.key == 'enter':
            self.step(self.env.actions.done)
            return


if __name__ == '__main__':
    states = []
    actions = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--env",
                        help="gym environment to load",
                        default='MiniGrid-MultiRoom-N6-v0')
    parser.add_argument("--seed",
                        type=int,
                        help="random seed to generate the environment with",
                        default=-1)
    parser.add_argument("--tile_size",
                        type=int,
                        help="size at which to render tiles",
                        default=32)
    parser.add_argument('--agent_view',
                        default=False,
                        help="draw the agent sees (partially observable view)",
                        action='store_true')

    args = parser.parse_args()

    env = gym.make(args.env_name)

    # if args.agent_view:
    #     env = RGBImgPartialObsWrapper(env)
    #     env = ImgObsWrapper(env)

    window = DemosWindow('gym_minigrid - ' + args.env, env, args)

    window.reset()

    # Blocking event loop
    window.show(block=True)
