import gym
import gym_minigrid
import numpy as np
import cv2

TILE_PIXELS = 32
TILE_SIZE = 32
DOWNSAMPLE_SIZE = 2  # how many times to downsample image before feeding to CNN


class CustomMGEnv(object):
    def __init__(self, env):
        self.__env = env

    def __getattr__(self, attr):
        if attr == '_CustomMGEnv__env':
            object.__getattr__(self, attr)
        else:
            return getattr(self.__env, attr)

    def __setattr__(self, attr, val):
        if attr == '_CustomMGEnv__env':
            object.__setattr__(self, attr, val)
        else:
            return setattr(self.__env, attr, val)

    def reset(self):
        ob = self.__env.reset()
        full_img, observable_img = self.render('rgb_array',
                                               tile_size=TILE_SIZE)
        ob['full_res_observable_img'] = self.preprocess_img(
            observable_img,
            (self.__env.height * TILE_SIZE, self.__env.width * TILE_SIZE),
            (self.__env.height * TILE_SIZE // DOWNSAMPLE_SIZE,
             self.__env.width * TILE_SIZE // DOWNSAMPLE_SIZE))
        return ob

    def step(self, act):
        ob, rwd, done, info = self.__env.step(act)
        full_img, observable_img = self.render('rgb_array',
                                               tile_size=TILE_SIZE)
        ob['full_res_observable_img'] = self.preprocess_img(
            observable_img,
            (self.__env.height * TILE_SIZE, self.__env.width * TILE_SIZE),
            (self.__env.height * TILE_SIZE // DOWNSAMPLE_SIZE,
             self.__env.width * TILE_SIZE // DOWNSAMPLE_SIZE))
        return ob, rwd, done, info

    def preprocess_img(self,
                       img,
                       max_shape,
                       output_shape,
                       color=[147, 147, 147]):
        img = img.copy()
        max_height = max_shape[0]
        max_width = max_shape[1]
        output_height = output_shape[0]
        output_width = output_shape[1]
        height = img.shape[0]
        width = img.shape[1]
        if height > max_height and width > max_width:
            print("img.shape: ", img.shape)
            print("max_shape: ", max_shape)
            img = cv2.resize(img, (max_width, max_height))

        elif height > max_height and width <= max_width:
            print("img.shape: ", img.shape)
            print("max_shape: ", max_shape)
            img = cv2.resize(img, (width, max_height))
            width_diff = max_width - width
            left = np.floor(width_diff / 2)
            right = np.ceil(width_diff / 2)
            img = cv2.copyMakeBorder(img,
                                     0,
                                     0,
                                     int(left),
                                     int(right),
                                     cv2.BORDER_CONSTANT,
                                     value=color)

        elif height <= max_height and width > max_width:
            print("img.shape: ", img.shape)
            print("max_shape: ", max_shape)
            img = cv2.resize(img, (max_width, height))
            height_diff = max_height - height
            bottom = np.floor(height_diff / 2)
            top = np.ceil(height_diff / 2)
            img = cv2.copyMakeBorder(img,
                                     int(top),
                                     int(bottom),
                                     0,
                                     0,
                                     cv2.BORDER_CONSTANT,
                                     value=color)

        else:
            width_diff = max_width - width
            left = np.floor(width_diff / 2)
            right = np.ceil(width_diff / 2)
            height_diff = max_height - height
            bottom = np.floor(height_diff / 2)
            top = np.ceil(height_diff / 2)
            img = cv2.copyMakeBorder(img,
                                     int(top),
                                     int(bottom),
                                     int(left),
                                     int(right),
                                     cv2.BORDER_CONSTANT,
                                     value=color)

        img = cv2.resize(img, (output_width, output_height))
        img = img.transpose(2, 0, 1)

        return img

    def render(self,
               mode='human',
               close=False,
               highlight=True,
               tile_size=TILE_PIXELS):
        """
           Render the whole-grid human view
           """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (
            self.agent_view_size - 1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height),
                                  dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None)

        nonzero_inds = highlight_mask.nonzero()
        xmin, xmax = nonzero_inds[0].min(), nonzero_inds[0].max()
        ymin, ymax = nonzero_inds[1].min(), nonzero_inds[1].max()
        observable_img = img[ymin * tile_size:(ymax + 1) * tile_size,
                             xmin * tile_size:(xmax + 1) * tile_size]

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img, observable_img


def make_env(env_key, seed=None):
    env = CustomMGEnv(gym.make(env_key))
    env.seed(seed)
    return env
