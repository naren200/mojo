# mojo

This repository contains the Mojo robot project with LeRobot integration for robot learning and control.

For demo and hackster blog: [https://www.hackster.io/josue-tristan/robot-cup-stacker-sometimes-thrower-36bd91](https://www.hackster.io/josue-tristan/robot-cup-stacker-sometimes-thrower-36bd91)

## Training and Evaluation with LeRobot-Mojo

The `lerobot-mojo` directory contains the LeRobot framework for training and evaluating robotic policies using imitation learning and reinforcement learning.

### Installation

#### Option 1: Docker Installation (Recommended for Jetson Thor)

The easiest way to get started, especially on **NVIDIA Jetson Thor** devices, is using Docker. This approach provides a pre-configured environment with all dependencies.

1. **Prerequisites:**
   - Docker and Docker Compose installed
   - NVIDIA Docker runtime (nvidia-docker2)
   - Base image pulled: `docker pull johnnync/lerobot:r38.2.aarch64-cu130-24.04`

2. **Set up environment variables:**
   Create a `.env` file in the `lerobot-mojo` directory or export these variables:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   export HF_USER=your_huggingface_username
   export DISPLAY=:0  # For GUI applications
   ```

3. **Build and run with Docker Compose:**
   ```bash
   cd lerobot-mojo
   docker-compose build
   docker-compose up -d
   ```

4. **Enter the container:**
   ```bash
   docker exec -it lerobot-mojo bash
   ```

The Docker setup includes:
- PyTorch 2.9.0 + CUDA 13.0 (optimized for Jetson Thor)
- LeRobot with Feetech motor support
- Orbbec camera SDK (pyorbbecsdk) built from source
- GPU acceleration with NVIDIA runtime
- USB device access for robot hardware
- HuggingFace cache persistence

#### Option 2: Manual Installation

1. Navigate to the lerobot-mojo directory:
```bash
cd lerobot-mojo
```

2. Create a virtual environment with Python 3.10:
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

3. Install ffmpeg (required for video processing):
```bash
conda install ffmpeg -c conda-forge
```

4. Install LeRobot:
```bash
pip install -e .
```

5. (Optional) Install simulation environments:
```bash
pip install -e ".[aloha, pusht]"
```

6. (Optional) Set up Weights & Biases for experiment tracking:
```bash
wandb login
```

### Training a Policy

**Note:** If using Docker, make sure you're inside the container first:
```bash
docker exec -it lerobot-mojo bash
```

All subsequent commands should be run from within the container or your activated virtual environment.

#### Basic Training Example

Train a policy using the command-line script:

```bash
python lerobot/scripts/train.py \
    --policy.name=diffusion \
    --env.type=pusht \
    --dataset.repo_id=lerobot/pusht \
    --batch_size=64 \
    --steps=5000 \
    --log_freq=100 \
    --save_freq=1000
```

#### Training with Weights & Biases Logging

Enable wandb for tracking training metrics:

```bash
python lerobot/scripts/train.py \
    --policy.name=diffusion \
    --env.type=pusht \
    --dataset.repo_id=lerobot/pusht \
    --wandb.enable=true \
    --wandb.project=my_robot_project
```

#### Training from a Configuration

Reproduce state-of-the-art results using pretrained configurations:

```bash
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```

#### Python Training Example

For more control, train directly from Python code (see `examples/3_train_policy.py`):

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load dataset
dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

# Create policy
dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)

# Train
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
# ... training loop ...
policy.save_pretrained("outputs/train/my_policy")
```

### Evaluating a Policy

#### Evaluate a Pretrained Model from HuggingFace Hub

```bash
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

#### Evaluate a Local Checkpoint

After training, evaluate your model checkpoints:

```bash
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/my_policy/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=100
```

#### Python Evaluation Example

Evaluate from Python code (see `examples/2_evaluate_pretrained_policy.py`):

```python
import gymnasium as gym
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load policy
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")

# Create environment
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)

# Run evaluation
policy.reset()
observation, info = env.reset(seed=42)

while not done:
    # Prepare observation
    state = torch.from_numpy(observation["agent_pos"]).to(torch.float32)
    image = torch.from_numpy(observation["pixels"]).to(torch.float32) / 255

    # Predict action
    with torch.inference_mode():
        action = policy.select_action({"observation.state": state, "observation.image": image})

    # Step environment
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
```

### Dataset Management

#### Visualize a Dataset

View episodes from a dataset:

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

Visualize a local dataset:

```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```

### Key Configuration Options

#### Training Configuration

- `--policy.name`: Policy architecture (diffusion, act, tdmpc, vqbet)
- `--dataset.repo_id`: HuggingFace dataset ID (e.g., lerobot/pusht)
- `--batch_size`: Training batch size (default: 64)
- `--steps`: Total training steps (default: 100000)
- `--log_freq`: Logging frequency in steps (default: 250)
- `--save_freq`: Checkpoint save frequency (default: 10000)
- `--eval_freq`: Evaluation frequency during training (default: 10000)
- `--policy.device`: Device to use (cuda, cpu)
- `--wandb.enable`: Enable Weights & Biases logging (default: false)

#### Evaluation Configuration

- `--policy.path`: Path to pretrained model (HuggingFace hub or local)
- `--env.type`: Environment type (pusht, aloha, xarm)
- `--eval.batch_size`: Number of parallel environments (default: 10)
- `--eval.n_episodes`: Number of episodes to evaluate (default: 50)
- `--policy.use_amp`: Use automatic mixed precision (default: true)

### Output Directory Structure

Training outputs are saved to:
```
outputs/train/
└── YYYY-MM-DD/
    └── HH-MM-SS_policy_name/
        ├── checkpoints/
        │   ├── 001000/
        │   │   └── pretrained_model/
        │   └── last/
        │       └── pretrained_model/
        └── eval/
            └── videos_step_XXXXX/
```

### Examples

The `examples/` directory contains demonstration scripts:

- `1_load_lerobot_dataset.py` - Load and inspect datasets
- `2_evaluate_pretrained_policy.py` - Evaluate pretrained models
- `3_train_policy.py` - Train a policy from scratch
- `advanced/` - Advanced topics like image transforms and validation loss

### Additional Resources

For more details, refer to the [lerobot-mojo README](lerobot-mojo/README.md) and the official [LeRobot documentation](https://github.com/huggingface/lerobot).
