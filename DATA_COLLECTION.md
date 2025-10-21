# Data Collection
For data collection, we need to calibrate the motors and collect data. This calibration is essential because it allows a neural network trained on one SO-10x robot to work on another.

Before proceeding, checkout to 
```bash
git -C lerobot-mojo checkout orbbec
```

## Hardware - Motors

### Motor Connections
1) Find motor ports for follower and leader. 

```bash
python lerobot-mojo/lerobot/scripts/find_motors_bus_port.py
```

2) Update the ports of follower and leader arm.
under the SO101RobotConfig class (`lerobot-mojo/lerobot/common/robot_devices/robots/configs.py`).
```bash
leader_arms: dict[str, MotorsBusConfig] = field(
    default_factory=lambda: {
        "main": FeetechMotorsBusConfig(
            port="/dev/ttyACM0",  <-- UPDATE HERE
follower_arms: dict[str, MotorsBusConfig] = field(
    default_factory=lambda: {
        "main": FeetechMotorsBusConfig(
            port="/dev//dev/ttyACM1",  <-- UPDATE HERE
```



##### Troubleshooting:
1) Make sure ports has required access
```bash
la -al
sudo chmod 666 /dev/ttyACM*
```

2) Make sure to check the IDs are not reset due to overload or over heating issues. Check the assigned ID values if motors are not found.
```bash
python lerobot-mojo/lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM0 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```



### Manual Calibration
Ensure that the leader and follower arms have the same position values when they are in the same physical position.  

##### Manual calibration of Follower arm

```bash
python lerobot-mojo/lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```


##### Manual calibration of leader arm

```bash
python lerobot-mojo/lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```


**IMPORTANT**: If you need to recalibrate the robotic arm, delete the `~/lerobot/.cache/huggingface/calibration/so101` folder.

### Teleoperate
`Simple teleop` Then you are ready to teleoperate your robot! Run this simple script (it won't connect and display the cameras):
```bash
python lerobot-mojo/lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

## Hardware - Camera


### Find Camera's index
1) To save the camera's data to find the index number for the respective camera's
```bash
python lerobot-mojo/lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

2) Update the camera's index for follower and leader arm.
```bash
cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "laptop": OpenCVCameraConfig(
            camera_index=0,             ##### UPDATE HEARE
        
        "phone": OpenCVCameraConfig(
            camera_index=1,             ##### UPDATE HEARE
            fps=30,
```

### Teleoperate
```bash
python lerobot-mojo/lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=teleoperate \
  --control.display_data=true
```

## Record the dataset

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/so101_test \
  --control.tags='["so101","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.display_data=true \
  --control.push_to_hub=false
```


## Visualize the dataset
```bash
echo ${HF_USER}/so101_test  
```
If you didn't upload with `--control.push_to_hub=false`, you can also visualize it locally with:

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/so101_test \
```
