# MultiorePerformancePrediction

## Tool I used
- Benchmark rodinia: https://github.com/yuhc/gpu-rodinia
- Benchmark Parsec 3.0
- Standard linux profiler: perf

## Data Collection
You can skip this step as I have put all the generated data in folder data/training_data.

But if you want to generate on your own, here are the steps:
Fot each host, go to folder `MultiorePerformancePrediction/scripts`
Set up by running
```
module load python-3.12
module load gcc-12.2

cd /home/.../MultiorePerformancePrediction/benchmark/parsec-benchmark
chmod +x env.sh
source env.sh
```

And then generating training data by running benchmark
```
cd /home/.../MultiorePerformancePrediction/scripts
nohup ./run_benchmark.sh 2>&1 > run_benchmark.txt &
tail -f run_benchmark.txt
```

## Data Analysis
The analysis script is `Analysis.py`. Remember to change the `directory` to `.../data/training_data` folder.

A python colab and its html version are also provided for result demonstration.
