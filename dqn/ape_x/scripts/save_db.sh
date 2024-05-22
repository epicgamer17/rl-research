#!/bin/bash

host=$1
port=$2
user=$3
password=$4
location=$5

uri="mongodb://$host:$port"
db="model_weights"

mongoexport --uri="$uri" --db="$db" --user="$user" --password="$password" --archive="$location" --gzip