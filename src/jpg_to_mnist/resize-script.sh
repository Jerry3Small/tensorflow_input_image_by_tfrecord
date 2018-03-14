#!/bin/bash

# simple script for resizing images in all class directories
# also reformats everything from whatever to png

if [ `ls ./testing_images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ./testing_images/*/*.jpg; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

#if [ `ls ./test_images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
#  echo hi
#  for file in ./test_images/*/*.png; do
#    convert "$file" -resize 28x28\! "${file%.*}.png"
#    file "$file" #uncomment for testing
#  done
#fi

if [ `ls ./training_images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ./training_images/*/*.jpg; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

#if [ `ls ./training_images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
#  echo hi
#  for file in ./training_images/*/*.png; do
#    convert "$file" -resize 28x28\! "${file%.*}.png"
#    file "$file" #uncomment for testing
#  done
#fi
