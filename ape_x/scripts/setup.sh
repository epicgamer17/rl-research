#!/bin/bash

# script to setup the environment for ape-x on open-gpu-x

# install capnp from source
install_capnp() {
  # don't install if already installed

  if [ -f "$HOME/.local/bin/capnp" ]; then
    echo "capnp already installed"
    return
  fi

  curl -O https://capnproto.org/capnproto-c++-1.0.2.tar.gz
  tar zxf capnproto-c++-1.0.2.tar.gz
  rm capnproto-c++-1.0.2.tar.gz
  cd capnproto-c++-1.0.2
  ./configure --prefix="$HOME/.local"
  make -j6 check
  make install
  rm -rf capnproto-c++-1.0.2
}

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
  if [ $CONDA_SHLVL -gt 0 ]; then
    for i in $(seq 1 $CONDA_SHLVL); do
      conda deactivate
    done 
  fi

  pip freeze | xargs pip uninstall -y

  for i in $(conda env list | grep -v "#" | grep -v "base" | awk '{print $1}'); do
    echo "deleting conda environment: $i"
    conda env remove -n $i
  done
}

echo "This script will delete your ~/.cache directory, install capnp from source, completely wipe out your default python installation and conda installations to make disk space and setup a new conda environment for ape-x."
echo
echo "It is also recommended to run this script on one of the open-gpu-x servers as they are much faster than mimi."
echo
read "Continue? (y/n) " out

if [ $out != "y" ]; then
  echo "Exiting..."
  exit 1
fi

# deleting .cache to make space

echo "Deleting ~/.cache to make space..."
rm -rf $HOME/.cache

echo "Installing capnp from source..."
install_capnp

echo "Setting up path..."
setup_path

echo "Are you sure you want to destroy default python environment and conda environments? This action is irreversible"
echo
read "Continue? (y/n) " out

if [ $out != "y" ]; then
  echo "Exiting..."
  exit 1
fi

echo "destroying default python environment..."
destroy_default_python_env

echo "Creating new conda environment..."

conda env create -f environment.yml
conda activate ml