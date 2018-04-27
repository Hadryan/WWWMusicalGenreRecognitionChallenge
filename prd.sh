#!/bin/bash

cd /www
for i in $(seq 0 9)
	do
		nohup python get_spectrogram.py run "$i" > "nohup.$i.out" &
	done

TEN=10
LIM=1
while true
	do echo "checking"
	NUMJOB=$(pgrep python | wc -l)
	if [ "$NUMJOB" -gt "$LIM" ];
		then echo "[$(NUMBJOB)] jobs are ongoing [$(date)]"
		sleep 10
	else
		echo "start training [$(date)]"
		python predict.py run --gpu_id "0"

	fi
	done


