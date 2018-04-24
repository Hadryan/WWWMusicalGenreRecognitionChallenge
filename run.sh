#!/bin/bash

docker pull minzwon/dl4mir:gpu-py2
docker run -v `pwd`/data/crowdai_fma_test:/crowdai-payload  -v `pwd`:/www -v /tmp:/tmp -it minzwon/dl4mir:gpu-py2  --name wwwmusicalgenrerecognitionchallenge_minzwon_test /www/prd.sh
docker cp wwwmusicalgenrerecognitionchallenge_minzwon_test:/tmp/output.csv .
