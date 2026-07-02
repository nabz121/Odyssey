# CIFAR10 Flow Matching: Reproduction Guide

This guide outlines the complete pipeline for reproducing both **Unconditional** and **Conditional** Flow Matching UNet models on the CIFAR10 dataset using the NUS SoC GPU cluster. Training is conducted on the cluster nodes, while sampling and image generation are handled via Google Colab.

---

## Step 1: Installing Miniconda

Before setting up the environment, ensure you are in your home directory on an active SoC compute node (e.g., `xlogin1`).

1. Navigate to your home directory:

```bash
cd ~
```

*Your terminal prompt should display:* `username@xlogin1:~$`

2. Download and run the Miniconda installation script:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

3. Verify your installation by listing your directory contents:

```bash
ls
```

*You should see both the installer script and the active directory:* `miniconda3` and `Miniconda3-latest-Linux-x86_64.sh`.

---

## Step 2: Cloning the Repository

Retrieve the core source code for the Flow Matching framework into your home directory:

```bash
git clone https://github.com/facebookresearch/flow_matching.git
```

---

## Step 3: Creating the Conda Environment

Set up a dedicated environment named `flow_matching` to install dependencies (PyTorch, Torchvision, etc.) required by the core framework.

1. Source your bash configuration to register the `conda` command:

```bash
source ~/.bashrc
```

2. Create and activate the target Python environment:

```bash
conda create --name flow_matching python=3.9 -y
conda activate flow_matching
```

*Your prompt will update to show the environment state:* `(flow_matching) username@xlogin1:~$`

3. Navigate to the image examples subfolder and install the required dependencies:

```bash
cd ~/flow_matching/examples/image
pip install -r requirements.txt
```

4. Deactivate the environment so you can safely hand off execution to the Slurm scheduler:

```bash
conda deactivate
```

*Verification prompt:* `username@xlogin1:~/flow_matching/examples/image$`

---

## Step 4: Submitting and Monitoring the Training Job

To train the model on the cluster GPUs, utilize the provided Slurm allocation script located within your `cifar_10_models` folder.

1. Ensure `run_unconditional.slurm` is copied into `~/flow_matching/examples/image`.
2. Submit the job and track its hardware allocation status:

```bash
# Submit the training task to the cluster queue
sbatch run_unconditional.slurm

# Monitor active job status
squeue -u $USER

# Review resource allocation and metrics
sacct --format=JobID,State,Elapsed,MaxRSS,ExitCode

# Stream live training progress (epochs, iterations, loss metrics)
tail -f cifar_train_2.log
```

---

## Step 5: Sampling and Inference via Google Colab

Once training completes, extract the model weights and runtime parameters from your output configuration directory (`~/flow_matching/examples/image/output_dir`) using `scp`.

### Assets & Artifacts

- **Target Folder:** Save `checkpoint-99.pth` and `args.json` inside a Google Drive folder named `flow_matching_project`.
- **Pre-trained Artifacts:** [Google Drive Shared Folder Link](https://drive.google.com/drive/folders/1Riyho0iFLvQvXjfFP8dK3SP4WL3P_GOk?usp=sharing)
- **Inference Pipeline:** [Google Colab Notebook Link](https://colab.research.google.com/drive/1ysh4Ac02lP-ZJzcl7Pd0bMNHnC5vMoEM?usp=sharing)

The inference notebook mounts your Google Drive, references the structural configuration, and leverages a custom **Euler Method Sampling** path trajectory to generate synthetic evaluation images.

---

This completes the comprehensive guide for replicating the Unconditional UNet framework.
