# An optimized ape_x implementation

Setup:

1. ssh into a fast SOCS server (e.g. open-gpu-n, NOT mimi!!) and run the scripts/setup.sh script
2. Install mongodb with the scripts/mongo_install.sh script
3. use `ssh_keygen -t ed25519` with all default options to generate a set of ssh keys to use for mimi to mimi ssh "hopping"
4. do `touch ~/.ssh/authorized_keys` then `cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys` to add authorize your key
5. run ./scripts/build.sh to build the go binaries (go conveniently comes preinstalled on mimi)
