#!/bin/bash

host=$1
port=$2
user=$3
password=$4
dir=$3

uri="mongodb://$host:$port"
db="model_weights"

mongoexport --uri="$uri" --db="$db" --user="$user" --password="$password" --collection="fs.files" --out="$dir"
mongoexport --uri="$uri" --db="$db" --user="$user" --password="$password" --collection="fs.chunks" --out="$dir"
mongoexport --uri="$uri" --db="$db" --user="$user" --password="$password" --collection="ids" --out="$dir"