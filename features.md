# Features for Lerobot

## Task 1, Merge torque feedback on SO101 arm to Lekiwi
1. on branch named so101_torque_feedback, I developed a feature to provide torque feedback from follower motors to leader motors, so when the follower motor are stuck, the operator will feel the resistance on the leader arm. This feature worked on that branch. The repo could be found at ~/Work/repo_tmp/lerobot/. The following files are changes in that repo:
 docs/manual/Feetech Bus Servo Memory Table(2025-11-11 21_46_44).xls
+232 −0  examples/so101/so101_control.sh
+12 −2  src/lerobot/robots/so101_follower/so101_follower.py
+55 −6  src/lerobot/scripts/lerobot_teleoperate.py
+43 −3  src/lerobot/teleoperators/so101_leader/so101_leader.py
+138 −0  src/lerobot/teleoperators/torque_feedback.py
+3 −3  src/lerobot/utils/visualization_utils.py

2. add this torque feedback feature on LeKiwi robot, under the branch named Lekiwi_v2. Lekiwi robot use the same SO101 arm, on a mobile platform. I want to have the torque feedback feature when teleoperate Lekiwi with the leader arm. 