#!/bin/bash

setup_file="$HOME/mongodb/mongodb_setup.js"
password_file="$HOME/mongodb/mongodb_admin_password"
mongo_port=5553
# check existing directories

if [ -d "$HOME/mongodb" ]; then
    echo "$HOME/mongodb directory already exists. Delete? (y/n)"
    read delete
    if [ $delete == "y" ]; then
        rm -r "$HOME/mongodb"
    else
        echo "Exiting..."
        exit 0
    fi
fi

install_mongodb() {
  echo "checking if MongoDB is already installed"

  if command -v mongod &> /dev/null
  then
      echo "MongoDB already installed"
  else
      echo "Installing mongodb"
      wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-7.0.8.tgz
      tar xzvf mongodb-linux-x86_64-ubuntu2204-7.0.8.tgz
      cp mongodb-linux-x86_64-ubuntu2204-7.0.8/bin/* ~/.local/bin
      rm mongodb-linux-x86_64-ubuntu2204-7.0.8.tgz
      rm -r mongodb-linux-x86_64-ubuntu2204-7.0.8

      chmod +x ~/.local/bin/mongod
  fi
}

install_mongosh() {
  echo "checking if mongosh is already installed"
  if command -v mongosh &> /dev/null
  then
    echo "mongosh already installed"
  else
    echo "Installing mongosh"
    wget https://downloads.mongodb.com/compass/mongosh-2.2.3-linux-x64.tgz
    tar xzvf mongosh-2.2.3-linux-x64.tgz
    cp mongosh-2.2.3-linux-x64/bin/mongosh ~/.local/bin
    cp mongosh-2.2.3-linux-x64/bin/mongosh_crypt_v1.so ~/.local/lib
    rm mongosh-2.2.3-linux-x64.tgz
    rm -r mongosh-2.2.3-linux-x64

    chmod +x ~/.local/bin/mongosh
  fi
}

create_admin_password() {
  hexdump -vn16 -e'4/4 "%08X" 1 "\n"' /dev/urandom > $password_file
  chmod 600 $password_file
}


mkdir "$HOME/mongodb"
mkdir "$HOME/mongodb/data"
mkdir "$HOME/mongodb/logs"

install_mongodb
install_mongosh

create_admin_password
touch "$setup_file"
echo "mongoPassword = \"$(cat $password_file)\";" > $setup_file
echo $(cat ./scripts/mongo_setup_template.js) >> $setup_file

mongod --fork --dbpath $HOME/mongodb/data --logpath $HOME/mongodb/logs/mongod.log 
mongosh --file $setup_file
echo "MongoDB setup complete, shutting down database"
mongod --shutdown --dbpath $HOME/mongodb/data



echo "use the following command to start the database:"
echo "mongod --dbpath $HOME/mongodb/data --logpath $HOME/mongodb/logs/mongod.log --port $mongo_port --auth --bind_ip_all"

echo "use the following command to connect to the database:"
echo "mongosh --port $mongo_port --authenticationDatabase admin -u ezra -p \$(cat $password_file)"

echo "use the following command to connect remotely to the database:"
echo "mongosh $(hostname -f) --port $mongo_port --authenticationDatabase admin -u ezra -p \$(cat $password_file)"