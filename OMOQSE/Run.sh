#!/bin/bash
# Runs the first input argument in Python 3.6 within a virtual environment
echo "Running python program in a virtual environment"

if [[ -d "venv_fist2" ]]; then
  echo "venv_fist2 exists"
  source ./venv_fist2/bin/activate
  echo "venv_fist2 activated"
  # pip3 install --upgrade tqdm
  # pip3 list
  #pip list
  nvcc --version
else
  echo "venv_fist2 doesn't exist"
  virtualenv --system-site-packages -p python3.6 ./venv_fist2
  source ./venv_fist2/bin/activate
  pip3 install --upgrade pip
  #Install required libraries
  pip3 install --upgrade numpy
  pip3 install --upgrade scipy  #would be skipped if pip3 install scipy
  pip3 install --upgrade pandas
  pip3 install --upgrade tqdm
  # pip3 install --upgrade torch torchvision
  #Install an older version of pytorch to align with the installed version of CUDA
  pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
  # pip3 install --upgrade torchaudio  #Need This for Feature Generation, but it breaks pytorch for the current cuda driver installed.
  pip3 install --upgrade matplotlib
  pip3 install --upgrade librosa
  pip3 install --upgrade pickle
  sudo apt-get install python3.6-tk #tkinter
  pip3 list
fi

# python3.6 CNN_example.py
python3.6 "$1"

deactivate
