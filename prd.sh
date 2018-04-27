#!/bin/bash

cd /www
for i in $(seq 0 9)
	do
		nohup python get_spectrogram.py run "$i" > "nohup.$i.out" &
	done
python predict.py run --gpu_id "0"

