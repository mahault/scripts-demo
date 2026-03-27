from haif_robot_control.robot_interfaces.lerobot_arm import LerobotArm
from haif_robot_control.robot_interfaces.px100_arm import Px100Arm
from haif_robot_control.robot_interfaces.hackerbot_arm import HackerbotArm
from haif_robot_control.robot_interfaces.carlito_arm import CarlitoArm

ROBOT_INFO = {
    'px100': {
        'urdf': "robot_description/px100/px100.urdf",
        'config_np': "configs/px100.yaml",
        'config_jax': "configs/jax/px100_jax.yaml",
        'port': "/dev/ttyUSB0",
        'baudrate': 115200,
        'n_joints': 4,
        'class_name': Px100Arm, 
    },
    'lerobot': {
        'urdf': "robot_description/SO101/lerobot.urdf",
        'config_np': "configs/lerobot.yaml",
        'config_jax': "configs/jax/lerobot_jax.yaml",
        'port': "/dev/ttyACM0",
        'baudrate': None,
        'n_joints': 5,
        'class_name': LerobotArm
    },
    'hackerbot': {
        "urdf": "robot_description/hackerbot/mycobot_280_arduino/hackerbot_arm.urdf",
        "urdf_w_world": "robot_description/hackerbot/mycobot_280_arduino/mycobot_280_arduino_original_w_world.urdf",
        "config_np": "configs/hackerbot.yaml",
        "config_jax": "configs/jax/hackerbot_jax.yaml",
        'port': "/dev/ttyACM2",
        'baudrate': 115200,
        'n_joints': 6,
        'class_name': HackerbotArm
    },
    'carlito': {
        'urdf': "robot_description/carlito/carlito_arm/wx250s.urdf",
        'config_np': "configs/widowx250s.yaml",
        'config_jax': "configs/jax/widowx250s_jax.yaml",
        'port': "/dev/ttyACM3",
        'baudrate': 115200,
        'n_joints': 6,
        'class_name': CarlitoArm
    },
    'fetch': {
        'urdf': "robot_description/fetch/robots/fetch_onlyarm.urdf",
        'config_np': "configs/fetch_arm.yaml",
        'config_jax': "configs/jax/fetch_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None, # No real robot interface
    },
    'r1pro_left_arm': {
        'urdf': "robot_description/r1pro/r1pro.urdf",
        'config_np': None,
        'config_jax': "configs/jax/r1pro_left_arm_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None, # No real robot interface
    },
    'r1pro_dual_arm': {
        'urdf': "robot_description/r1pro/r1pro.urdf",
        'config_np': None,
        'config_jax': "configs/jax/r1pro_dual_arm_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None, # No real robot interface
    },
    'panda': {
        'urdf': "robot_description/panda/panda.urdf",
        'config_np': "configs/panda_arm.yaml",
        'config_jax': "configs/jax/panda_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None, # No real robot interface
    },
    'widowx': {
        'urdf': "robot_description/widowx/widowx.urdf",
        'config_np': "configs/widowx.yaml",
        'config_jax': "configs/jax/widowx_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None, # No real robot interface
    },
    'jaco': {
        'urdf': "robot_description/jaco/jaco_wo_base.urdf",
        'config_np': "configs/jaco.yaml",
        'config_jax': "configs/jax/jaco_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None, # No real robot interface
    },
    '2dof': {
        'urdf': "robot_description/2dof.urdf",
        'config_np': "configs/2dof.yaml",
        'config_jax': "configs/jax/2dof_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None,
    },
    '3dof': {
        'urdf': "robot_description/3dof.urdf",
        'config_jax': "configs/jax/3dof_jax.yaml",
        'port': None,
        'baudrate': None,
        'class_name': None,
    }
}