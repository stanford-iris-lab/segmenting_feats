import gym
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from PIL import Image



env_name = 'assembly-v2-goal-observable'

# for shift in ['_distractor_medium', '_granite_table', '_metal1_table', '_cast_right', '_cast_left', '_darker', '_brighter', '']:
for shift in ['_distractor_medium']:
    e = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](model_name=f'/iris/u/kayburns/packages/metaworld_r3m/metaworld/envs/assets_v2/sawyer_xyz/sawyer_assembly_peg{shift}.xml')
    e = gym.ObservationWrapper(e)
    im = e.render()
    image = Image.fromarray(im)
    image.save(f'test{shift}.jpg')