#!/bin/bash

docker pull minzwon/dl4mir:gpu-py2
nvidia-docker run --rm -it -v `pwd`/data/crowdai_fma_test:/crowdai-payload\
	-v `pwd`:/www\
	-v /tmp:/tmp\
	minzwon/dl4mir:gpu-py2 /www/prd.sh
