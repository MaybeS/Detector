#!/bin/bash

# check docker installed
echo "=== Checking docker exists...";
docker=`which docker`
if [ -z $docker ]; then
  echo "It looks like the docker is not installed."
  exit 0
fi

# get variable with default value
# @Param {message}
# @Param {default}
get_variable () {
  local temp=''
  read -p "$1 [default: $2]: " temp
  if [ -z $temp ]; then
    echo "$2"
  else
    echo "$temp"
  fi
}

echo "=== SETUP === Just press [Enter] to use default value";
IMAGE_NAME=$( get_variable "App Image Name" "demo-app")
CONTAINER_NAME=$( get_variable "App Container Name" "demo-ssd")
PORT=$( get_variable "Database Port" "8080")

$docker build -q -t $IMAGE_NAME .
$docker run -d --name $CONTAINER_NAME -p $PORT:80 -e WORKERS_PER_CORE=".25" $IMAGE_NAME
