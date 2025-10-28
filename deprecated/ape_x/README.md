# A Pytorch RPC-based ape_x implementation

Setup:

1. ssh into a fast SOCS server (e.g. open-gpu-n, NOT mimi!!) and run the `scripts/setup.sh` script (from the ape_x directory)
2. use `ssh_keygen -t ed25519` with all default options to generate a set of ssh keys to use for mimi to mimi ssh "hopping"
3. run `touch ~/.ssh/authorized_keys` then `cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys` to add your key
4. run ./scripts/build.sh to build the go binaries (go conveniently comes preinstalled on mimi)
5. run `conda activate ml`