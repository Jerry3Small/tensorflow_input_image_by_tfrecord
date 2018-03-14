#!/bin/bash

rename 's:(.+)\s(\[.+\])-(\d)-(\d)\.jpeg:$1_$3_$4.jpg:' ./training_images/monster/*.jpeg
rename 's:(.+)\s(\[.+\])-(\d)-(\d)\.jpeg:$1_$3_$4.jpg:' ./training_images/spell/*.jpeg
rename 's:(.+)\s(\[.+\])-(\d)-(\d)\.jpeg:$1_$3_$4.jpg:' ./training_images/trap/*.jpeg

rename 'y/A-Z/a-z/' ./training_images/monster/*
rename 'y/A-Z/a-z/' ./training_images/spell/*
rename 'y/A-Z/a-z/' ./training_images/trap/*
rename 'y/A-Z/a-z/' ./testing_images/monster/*
rename 'y/A-Z/a-z/' ./testing_images/spell/*
rename 'y/A-Z/a-z/' ./testing_images/trap/*


