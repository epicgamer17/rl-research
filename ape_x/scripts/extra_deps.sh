#!/bin/bash

# extra dependencies for the ml environment not included in the environment.yml
conda activate ml
pip install zmq
pip install gymnasium[classic-control]
pip install moviepy