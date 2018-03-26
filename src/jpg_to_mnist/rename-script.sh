#!/bin/bash

rename 's:(.+)\s(\[.+\])-(\d)-(\d+)\.jpeg:$1_$3_$4.jpg:' ./training_images/0/*.jpeg
rename 's:(.+)\s(\[.+\])-(\d)-(\d+)\.jpeg:$1_$3_$4.jpg:' ./training_images/1/*.jpeg
rename 's:(.+)\s(\[.+\])-(\d)-(\d+)\.jpeg:$1_$3_$4.jpg:' ./training_images/2/*.jpeg

rename 'y/A-Z/a-z/' ./training_images/0/*
rename 'y/A-Z/a-z/' ./training_images/1/*
rename 'y/A-Z/a-z/' ./training_images/2/*
rename 'y/A-Z/a-z/' ./testing_images/0/*
rename 'y/A-Z/a-z/' ./testing_images/1/*
rename 'y/A-Z/a-z/' ./testing_images/2/*


