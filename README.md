
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

