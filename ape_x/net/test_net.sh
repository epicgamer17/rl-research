#!/bin/bash

# script to test what computers are online and what each one is connected to

# see https://www.cs.mcgill.ca/docs/resources/

username=ehuang
file=data.log
timeout=1

online_opengpu=0
test_gpu_computers() {
  for i in $(seq 1 32);do
    host="$username@open-gpu-$i.cs.mcgill.ca"
    echo $host
    echo "$host ===============================" >> $file
    ssh -oStrictHostKeyChecking=no -oConnectTimeout=$timeout "$host" -f 'arp' >> $file
    if [ $? -eq 0 ]; then
      online_opengpu=$((online_opengpu+1))
    fi
    sleep 1
  done
}

online_lab_one=0
test_lab1_computers() {
  for i in $(seq 1 12);do
    host="$username@lab1-$i.cs.mcgill.ca"
    echo "$host ===============================" >> $file
    ssh -oStrictHostKeyChecking=no -oConnectTimeout=$timeout "$host" -f 'arp' >> $file
    if [ $? -eq 0 ]; then
      online_lab_one=$((online_lab_one+1))
    fi
    sleep 1
  done
}

online_lab_two=0
test_lab2_computers() {
  for i in $(seq 1 51);do
    host="$username@lab2-$i.cs.mcgill.ca"
    echo "$host ===============================" >> $file
    ssh -oStrictHostKeyChecking=no -oConnectTimeout=$timeout "$host" -f 'arp' >> $file
    if [ $? -eq 0 ]; then
      online_lab_two=$((online_lab_two+1))
    fi
    sleep 1
  done
}

echo "" > $file
test_gpu_computers
test_lab1_computers
test_lab2_computers

echo "online open-gpu: $online_opengpu" >> $file
echo "online lab1: $online_lab_one" >> $file
echo "online lab2: $online_lab_two" >> $file