from gym.envs.registration import register

register(
    id='LifeGate-v1',
    entry_point='LifeGate.envs:LifeGate')