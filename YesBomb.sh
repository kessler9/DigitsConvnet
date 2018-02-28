#!/bin/bash
if [[ ! -z $1 ]]; then
	for i in `seq $1`; do
		echo "STARTING YES #"$i
		yes &
	done
else
	echo "PLEASE SPECIFY CONCURRENT YES THREADS"
fi
