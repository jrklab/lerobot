from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

camera_config = {
    "front": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=30) # fps=15 might be more stable
}

robot_config = SO101FollowerConfig(
    port="/dev/ttyUSB1",
    id="follower_arm_1",
    cameras=camera_config
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyUSB0",
    id="leader_arm_1",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)