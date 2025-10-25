#!/bin/bash

# ==============================================================================
# Lerobot Control Script
#
# A wrapper to simplify common tasks for the Hugging Face Lerobot framework.
#
# Usage:
#   ./robot_control.sh [option]
#
# Options:
#   teleop      - Start the teleoperation interface to control the robot manually.
#   record      - Record training data by teleoperating the robot.
#   train       - Train a policy on a recorded dataset.
#   eval        - Evaluate a trained policy in the environment.
#
# ==============================================================================

# --- CONFIGURATION ------------------------------------------------------------
# TODO: Modify these variables to match your project setup.

# The name of your robot environment (e.g., xarm_teleop, allegro_hand_teleop)
ENV_NAME="SO101_ARM"

# The Hugging Face repository ID where your dataset is/will be stored.
DATASET_REPO_ID="jrkhf/so101_wrist_top_cameras_set_merged" # "jrkhf/so101_set_2"

# The Hugging Face repository ID where your dataset is/will be stored.
EVAL_REPO_ID="jrkhf/eval_so101"

# The Hugging Face repository ID where your trained policy is/will be stored.
POLICY_REPO_ID="jrkhf/so101_dual_cameras_act_policy_100k" # "lerobot/smolvla_base"

# --- TASK-SPECIFIC PARAMETERS -------------------------------------------------

# Number of episodes to record during a data collection session.
NUM_RECORD_EPISODES=20

# Number of training steps for the policy.
NUM_TRAIN_STEPS=20000

# Number of episodes to run during policy evaluation.
NUM_EVAL_EPISODES=10

# The name of the policy checkpoint to use for evaluation (e.g., "last", "step_10000").
EVAL_CHECKPOINT="last"

# ------------------------------------------------------------------------------
# SCRIPT LOGIC - Do not edit below this line unless you know what you are doing.
# ------------------------------------------------------------------------------

# Function to display usage information
usage() {
    echo "Usage: $0 {teleop|record|train|eval}"
    echo "  teleop:      Start teleoperation interface."
    echo "  record:      Record ${NUM_RECORD_EPISODES} episodes of training data to '${DATASET_REPO_ID}'."
    echo "  train:       Train a policy for ${NUM_TRAIN_STEPS} steps."
    echo "  eval:        Evaluate the '${EVAL_CHECKPOINT}' checkpoint of policy '${POLICY_REPO_ID}'."
}

# Check if an option was provided
if [ -z "$1" ]; then
    echo "Error: No option provided."
    usage
    exit 1
fi

# Main control logic based on the first argument
case "$1" in
    teleop)
        echo ">>> Starting teleoperation for environment: ${ENV_NAME}"
        lerobot-teleoperate \
            --robot.type=so101_follower \
            --robot.port=/dev/ttyUSB1 \
            --robot.id=follower_arm_1 \
            --robot.cameras="{top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, \
                              wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
            --teleop.type=so101_leader \
            --teleop.port=/dev/ttyUSB0 \
            --teleop.id=leader_arm_1 \
            --display_data=true
        ;;
    record)
        echo ">>> Starting data recording..."
        echo "    Environment:  ${ENV_NAME}"
        echo "    Dataset Repo: ${DATASET_REPO_ID}"
        echo "    Episodes:     ${NUM_RECORD_EPISODES}"
        lerobot-record \
            --robot.type=so101_follower \
            --robot.port=/dev/ttyUSB1 \
            --robot.id=follower_arm_1 \
            --robot.cameras="{top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, \
                              wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
            --teleop.type=so101_leader \
            --teleop.port=/dev/ttyUSB0 \
            --teleop.id=leader_arm_1 \
            --display_data=true \
            --dataset.repo_id=${DATASET_REPO_ID} \
            --dataset.num_episodes=${NUM_RECORD_EPISODES} \
            --dataset.single_task="Pick and drop lego block" \
            --dataset.episode_time_s=60 \
            --dataset.push_to_hub=True \
            --resume=false

        ;;
    train)
        echo ">>> Starting policy training..."
        echo "    Dataset Repo: ${DATASET_REPO_ID}"
        echo "    Policy Repo:  ${POLICY_REPO_ID}"
        echo "    Train Steps:  ${NUM_TRAIN_STEPS}"
        lerobot-train \
            --dataset.repo_id=${DATASET_REPO_ID} \
            --policy.type=act \
            --output_dir="outputs/train/act_so101_test" \
            --job_name=act_so101_test \
            --policy.device=cpu \
            --wandb.enable=false \
            --policy.repo_id=${POLICY_REPO_ID} \
            --batch_size=8 \
            --steps=${NUM_TRAIN_STEPS} \
            --save_freq=5000 \
            --dataset.image_transforms.enable=true \
            --dataset.image_transforms.random_order=true \
            --dataset.image_transforms.max_num_transforms=2 \
            --resume=false

        ;;
    eval)
        echo ">>> Starting policy evaluation..."
        echo "    Environment: ${ENV_NAME}"
        echo "    Policy Repo: ${POLICY_REPO_ID}"
        echo "    Eval Repo:  ${EVAL_REPO_ID}"
        echo "    Episodes:    ${NUM_EVAL_EPISODES}"
        lerobot-record \
            --robot.type=so101_follower \
            --robot.port=/dev/ttyUSB1 \
            --robot.cameras="{top: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, \
                            wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
            --robot.id=follower_arm_1 \
            --display_data=true \
            --dataset.repo_id=${EVAL_REPO_ID} \
            --dataset.single_task="Move lego block" \
            --dataset.num_episodes=${NUM_EVAL_EPISODES} \
            --dataset.episode_time_s=120 \
            --teleop.type=so101_leader \
            --teleop.port=/dev/ttyUSB0 \
            --teleop.id=leader_arm_1 \
            --policy.path=${POLICY_REPO_ID}
        ;;
    tfs_viz)
        echo ">>> Starting image transformation visualization..."
        echo "    Dataset Repo: ${DATASET_REPO_ID}"
        lerobot-imgtransform-viz \
            --repo_id=${DATASET_REPO_ID} \
            --image_transforms.enable=true \
            --image_transforms.max_num_transforms=3 \
            --image_transforms.random_order=false
        ;;
    merge_datasets)
        echo ">>> Merging datasets..."
        lerobot-edit-dataset \
            --repo_id=jrkhf/so101_wrist_top_cameras_set_merged \
            --operation.type=merge \
            --operation.repo_ids="['jrkhf/so101_wrist_top_cameras_set_2', 'jrkhf/so101_wrist_top_cameras_set_daytime']" \
            --push_to_hub=true \
            --operation.reset_frame_indices=true
        ;;
    *)
        echo "Error: Invalid option '$1'."
        usage
        exit 1
        ;;
esac

echo ">>> Task '$1' completed."
