**Status:** Archive (code is provided as-is, no updates expected)

# Safety Gym

Forked from the original `safety-gym` environment. 
To get started, make sure to checkout the original repo [here](https://github.com/openai/safety-gym). 

This repo is modified so that we can generate fixed env layouts for RL training. 
If we set the random seed of the env to be a fixed one, the env layout will be fixed. 
If we give the env multiple random seeds, the number of different layouts will be equal to the number of the seeds. 
Please checkout the tests in `saferl-envs/tests/test_safety_gym.py` to learn more about this behavior. 