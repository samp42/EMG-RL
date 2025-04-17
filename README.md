# EMG RL

Teaching a hand to move based on Electromyography (EMG) data using reinforcement learning.

See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) for Conda documentation.

To install the necessary packages, run:

```bash
conda create -n emg-rl python=3.12
conda activate emg-rl
pip install -r requirements.txt
```

# Setup

## Install MuJoCo

Download MuJoCo 2.1.0 following these instructions: https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da

(testing the Py example should fail here)

Then run (https://github.com/openai/mujoco-py/issues/291#issuecomment-1738943394):

```
sudo apt-get install patchelf
pip install Cython==3.0.0a10
```

## Install dexterous-gym

Clone the repo and navigate to its directory to run:

```
pip install -e .
```

## Install gym

```
pip install gym==0.15.3
```

## Install the repository environment

Navigate to the repo top-level and run:

```
pip install -e .
```

# Data

Download the DB5 data (with uncalibrated CyberGlove data) from https://ninapro.hevs.ch/instructions/DB5.html
Download the DB5 CyberGlove calibrated data (s68 to s77) from https://ninapro.hevs.ch/instructions/DB9.html

Copy the all the data into a "data" directory.
