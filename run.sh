#!/bin/bash

# export CONTAINER_NAME="wwwmusicalgenrerecognitionchallenge_minzwon"
chmod +x run.sh
chmod +x prd.sh

docker pull minzwon/dl4mir:gpu-py2
docker run --name $CONTAINER_NAME -v `pwd`/data/crowdai_fma_test:/crowdai-payload  -v `pwd`:/www -v /tmp:/tmp -it minzwon/dl4mir:gpu-py2  /www/prd.sh
docker cp $CONTAINER_NAME:/tmp/output.csv .
