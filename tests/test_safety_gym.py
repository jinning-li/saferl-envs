import unittest

import gym
import numpy as np
import safety_gym


class TestEnvLayout(unittest.TestCase):

    def test_fixed_layout_after_reset_env(self):
        '''Env layout should be the same for a fixed random seed after reset'''
        env_name = "Safexp-CarButton1-v0"
        seed = 123

        env1 = gym.make(env_name)
        env1.seed(seed)  # Set the random seed for the env to be a fixed one!
        obs1 = env1.reset()
        obs2 = env1.reset()
        np.testing.assert_almost_equal(obs1, obs2)

    def test_fixed_layout_multiple_envs_same_seed(self):
        '''Env layouts should be the same if initialize multiple envs
        with a fixed random seed'''
        env_name = "Safexp-CarButton1-v0"
        seed = 123

        env1 = gym.make(env_name)
        env1.seed(seed)  # Set the random seed for the env to be a fixed one!
        env2 = gym.make(env_name)
        env2.seed(seed)
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_almost_equal(obs1, obs2)

    def test_fixed_multiple_layouts(self):
        '''Set the number of different layouts'''
        env_name = "Safexp-CarButton1-v0"
        seed = 123
        num_different_layouts = 3

        env = gym.make(env_name)
        env.seed(seed)
        env.set_num_different_layouts(num_different_layouts)

        obs1 = env.reset()
        obs2 = env.reset()
        obs3 = env.reset()
        
        obs4 = env.reset()
        obs5 = env.reset()
        obs6 = env.reset()

        np.testing.assert_almost_equal(obs1, obs4)
        np.testing.assert_almost_equal(obs2, obs5)
        np.testing.assert_almost_equal(obs3, obs6)


if __name__ == '__main__':
    unittest.main()
