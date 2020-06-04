#!/bin/bash
# Runs the first input argument in Python 3.6 within a virtual environment
echo "Running python program in a virtual environment"

if [[ -d "venv" ]]; then
  echo "venv exists"
  source ./venv/bin/activate
  echo "venv activated"
  # pip3 list
  #pip list
else
  echo "venv doesn't exist"
  virtualenv --system-site-packages -p python3.6 ./venv
  source ./venv/bin/activate
  pip3 install --upgrade pip
  #Install required libraries
  pip3 install --upgrade numpy
  pip3 install --upgrade scipy  #would be skipped if pip3 install scipy
  pip3 install --upgrade pandas
  pip3 install --upgrade torch torchvision
  pip3 install --upgrade matplotlib
  pip3 install --upgrade librosa
  pip3 install --upgrade pickle
  sudo apt-get install python3.6-tk #tkinter
  pip3 list
fi

# python3.6 CNN_example.py
python3.6 "$1"

deactivate
