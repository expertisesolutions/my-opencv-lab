#!/bin/env sh

ffmpeg -i TownCentreXVID.mp4 -s 1920x1080 -b:v 512k -vcodec mpeg1video -acodec copy 1080p_TownCentreXVID.mp4
ffmpeg -i TownCentreXVID.mp4 -s 1280x720 -b:v 512k -vcodec mpeg1video -acodec copy 720p_TownCentreXVID.mp4
ffmpeg -i TownCentreXVID.mp4 -s 854x480 -b:v 512k -vcodec mpeg1video -acodec copy 480p_TownCentreXVID.mp4
ffmpeg -i TownCentreXVID.mp4 -s 640x360 -b:v 512k -vcodec mpeg1video -acodec copy 360p_TownCentreXVID.mp4

