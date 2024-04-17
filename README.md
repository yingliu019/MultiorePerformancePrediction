# MultiorePerformancePrediction

## Tool I used
- Benchmark rodinia: https://github.com/yuhc/gpu-rodinia
- Benchmark Parsec 3.0
- Standard linux profiler: perf

## Data Collection
Go to folder `MultiorePerformancePrediction/scripts`
For each host, set up by running
```
module load python-3.12
module load gcc-12.2

cd /home/yl3750/MultiorePerformancePrediction/benchmark/parsec-benchmark
chmod +x env.sh
source env.sh
```
And then generating raining data by running benchmark
```
cd /home/yl3750/MultiorePerformancePrediction/scripts
nohup ./run_benchmark.sh 2>&1 > run_benchmark.txt &
tail -f run_benchmark.txt
```

## Data Analysis
The analysis script is `Analysis.py`. Remember to change the directory to training data folder.
A python colab and its pdf version is also provided for result demonstration.
