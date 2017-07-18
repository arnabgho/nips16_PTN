#!/bin/bash
filename='/imgs/a360_e030.jpg'
folder='imgs/'
for file in $(ls $1);
do
  command= $(cp ${PWD}$filename $folder$file)
  eval $command
done
