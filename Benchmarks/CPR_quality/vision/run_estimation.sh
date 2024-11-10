#!/bin/bash

echo "Starting the vision benchmark"
python /home/cogems_nist/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/estimate.py 0 0 30

echo "Benchmark 1 is done"

sleep 20

python /home/cogems_nist/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/estimate.py 0 30 70

echo "Benchmark 2 is done"

sleep 20

python /home/cogems_nist/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/estimate.py 0 70 110

echo "Benchmark 3 is done"

sleep 20

python /home/cogems_nist/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/estimate.py 0 110

echo "Benchmark 4 is done"

sleep 20

python /home/cogems_nist/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/estimate.py 1 0

echo "Benchmark 5 is done"

sleep 20

python /home/cogems_nist/repos/EgoExoEMS/Benchmarks/CPR_quality/vision/estimate.py 2 0 

echo "Benchmark 6 is done"
