#!/bin/bash

curl -O https://capnproto.org/capnproto-c++-1.0.2.tar.gz
tar zxf capnproto-c++-1.0.2.tar.gz
rm capnproto-c++-1.0.2.tar.gz
cd capnproto-c++-1.0.2
./configure --prefix="$HOME/.local"
make -j6 check
make install