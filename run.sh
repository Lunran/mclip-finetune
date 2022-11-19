#!/bin/sh

if [ $# != 1 ]; then
    echo "Requires the ID of this run"
    exit 1
fi

rm -rf ~/.cache/wandb/ wandb wandb_mclip
#python src/$1/run.py run.id=$1
nohup python src/$1/run.py run.id=$1 >run.log 2>&1 &
