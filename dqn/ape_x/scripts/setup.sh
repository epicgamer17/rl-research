#!/bin/bash

# script to setup the environment for ape-x on open-gpu-x

# path setup
setup_path() {
  if [ ! -f "$HOME/.bashrc" ]; then
    touch "$HOME/.bashrc"
  fi

  echo "export PATH=$HOME/.local/bin:\$PATH" >> "$HOME/.bashrc"
  source ~/.bashrc
}

# destroy default python environment and conda environments
destroy_default_python_env() {
  if [ -n "$CONDA_SHLVL" ]; then
    if [ "$CONDA_SHLVL" -gt 0 ]; then
      for i in $(seq 1 $CONDA_SHLVL); do
        conda deactivate
      done 
    fi
  fi

  pip freeze | xargs pip uninstall -y

  for i in $(conda env list | grep -v "#" | grep -v "base" | awk '{print $1}'); do
    echo "deleting conda environment: $i"
    conda env remove -n $i
  done
}

echo "This script will delete your ~/.cache directory, completely wipe out your default python installation and conda installations to make disk space and setup a new conda environment for ape-x."
echo
echo "It is also recommended to run this script on one of the open-gpu-x servers as they are much faster than mimi."
echo
read -p "Continue? (y/n) " out

if [ $out != "y" ]; then
  echo "Exiting..."
  exit 1
fi

# deleting .cache to make space

echo "Deleting ~/.cache to make space..."
rm -rf $HOME/.cache

echo "Are you sure you want to destroy default python environment and conda environments? This action is irreversible"
echo
read -p "Continue? (y/n) " out

if [ $out != "y" ]; then
  echo "Exiting..."
  exit 1
fi

echo "destroying default python environment..."
destroy_default_python_env


echo "Creating new conda environment..."

conda env create -f ./scripts/environment.yml
conda init bash
conda activate ml

echo "installing additional dependencies"
pip install zmq
pip install gymnasium[classic-control]
pip install moviepy
pip install pymango