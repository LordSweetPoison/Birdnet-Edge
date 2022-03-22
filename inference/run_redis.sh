#!/bin/bash

# if redis is not installed, install and unzip redis
if [ ! -d redis-stable/src ]; then
    curl -O http://download.redis.io/redis-stable.tar.gz
    tar xvzf redis-stable.tar.gz
    rm redis-stable.tar.gz
fi

cd redis-stable

# make redis
make

# start redis server 
src/redis-server