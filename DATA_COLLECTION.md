
### 
Find motor ports for follower and leader. 

```bash
python lerobot-mojo/lerobot/scripts/find_motors_bus_port.py
```



### Calibration
Ensure that the leader and follower arms have the same position values when they are in the same physical position. This calibration is essential because it allows a neural network trained on one SO-10x robot to work on another. 

If you need to recalibrate the robotic arm, delete the `~/lerobot/.cache/huggingface/calibration/so101` folder.
