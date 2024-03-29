#!/bin/bash

username=ehuang
replay_server=open-gpu-32
learner=open-gpu-1
mimi=cs.mcgill.ca
replay_learner_port=5554
replay_actor_port=5555
learner_port=5556
timeout=1

# start replay server
start_replay() {
  echo "Starting replay server"
  ssh -oStrictHostKeyChecking=no -oConnectTimeout=$timeout $username@$replay_server.$mimi << EOF
  cd ~/rl-research/ape_x 
  conda activate ml 
  python3 distributed_replay_buffer.py $replay_learner_port $replay_actor_port &
EOF
}

# start learner server
start_learner() {
  echo "Starting learner"
  ssh -oStrictHostKeyChecking=no -oConnectTimeout=$timeout $username@$learner.$mimi << EOF
  cd ~/rl-research/ape_x 
  conda activate ml 
  python3 main_learner.py $learner_port $replay_server.$mimi $replay_learner_port &
EOF
}

# start an actor
start_actor() {
  ssh -oStrictHostKeyChecking=no -oConnectTimeout=$timeout username@$actor.$mimi << EOF
  cd ~/rl-research/ape_x 
  conda activate ml 
  python3 main_actor.py 0 $learner.$mimi $learner_port $replay_server.$mimi $replay_actor_port &
EOF
}

start_replay
start_learner

# start actors
for i in {2..31}; do
  if [ $i -eq 17 ]; then 
    continue
  fi

  actor="open-gpu-$i"
  echo "Starting actor on $actor"
  start_actor
done