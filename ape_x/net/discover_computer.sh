#!/bin/bash

# script to test what ip addresses correspond to what computers

file=hosts.log
for i in $(seq 1 255); do
  echo -n "10.69.38.$i " >> $file
  host 10.69.38.$i | awk "{print \$5}" >> $file
done

