# MultiorePerformancePrediction
This folder includes all the code for data generation and analysis. 

And it doesn't include the benchmark suite input data as they are too large. 

## Tool I used
- Benchmark rodinia: https://github.com/yuhc/gpu-rodinia
- Benchmark Parsec 3.0
- Standard linux profiler: perf
- computer servers: https://cims.nyu.edu/dynamic/systems/resources/computeservers/

## Data Collection
You can skip this step as I have put all the generated data in `.../data/training_data` folder.

But if you want to generate on your own, here are the steps:

First get the `benchmark.zip`, it has input data included.

Unzip the file `benchmark.zip` and overwrite the `.../benchmark` folder.

For each host, go to folder `MultiorePerformancePrediction/scripts`. Set up by running
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
The analysis script is `Analysis.py`. I used `python-3.12` and the package version is in `requirements.txt`.

Also change the `directory` in `Analysis.py` to where `training_data` exists.

A python colab and its html version are also provided for the result demonstration.

## Slides
https://docs.google.com/presentation/d/1DnRcN1lfxR6Gpr4GAZ_Nhiy444KQEuSxurn_R_yQXns/edit#slide=id.g2d046dbd239_0_763

