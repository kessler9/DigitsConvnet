#!/bin/bash

rm -f *.log && rm -f *.kesstest

TEST_FILES=`ls *relu*`

for test in $TEST_FILES; do
#	echo `echo $test | sed 's/.py/.log/'`
#	echo `echo $test | sed 's/.py/.kesstest/'`
	rm -rf MNIST_Model;
	wait $!;
	echo "FINISHED FLUSH";
	FILENAME=`echo $test | sed 's/.py//'`".log";
	echo "FILENAME: " $FILENAME;
	python $test > $FILENAME;
	wait $!;
	echo "FINISHED TRAINING";
	FILENAME=`echo $test | sed 's/.py//'`".kesstest";
        echo "FILENAME: " $FILENAME;
	python testall.py > $FILENAME;
	wait $!;
	echo "FINISHED TESTING";
done
