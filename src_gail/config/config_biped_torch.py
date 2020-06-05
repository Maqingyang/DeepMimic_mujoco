from os import getcwd

class Config(object):
    all_motions = [ 'run', 'walk']
    curr_path = getcwd()
    # motion = 'spinkick'
    motion = 'walk'
    env_name = "bipedal_2d_torch"

    motion_folder = '/mujoco/motions'
    xml_folder = '/mujoco/bipedal/envs/asset'

    mocap_path = "%s%s/biped_%s.txt"%(curr_path, motion_folder, motion)
    xml_path = "%s%s/%s.xml"%(curr_path, xml_folder, env_name)