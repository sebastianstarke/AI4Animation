#!/bin/bash
 # echo of all files in a directory

for file in *.stl
do
  name=${file%%[.]*}
  meshlabserver -i $file -o $name'.stl' -om vn -s chull-simple.mlx
  echo $name
done
