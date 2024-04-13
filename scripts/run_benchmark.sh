# nohup ./run_benchmark.sh 2>&1 > run_benchmark.txt &
# tail -f run_benchmark.txt

echo "==================Preparation (one time per host)====================="
echo "-----load modules-----"
module load python-3.12
module load gcc-12.2

echo "-----prepare parsec-----"
chmod +x /home/yl3750/MultiorePerformancePrediction/benchmark/parsec-benchmark/env.sh
source /home/yl3750/MultiorePerformancePrediction/benchmark/parsec-benchmark/env.sh

echo "==================Run Rodinia (one csv per line)====================="
echo "-----bfs-----"
python3 run_benchmark.py --prog=bfs --size=simsmall --input=../../data/bfs/graph4096.txt --bm=rodinia
python3 run_benchmark.py --prog=bfs --size=simmedium --input=../../data/bfs/graph65536.txt --bm=rodinia
python3 run_benchmark.py --prog=bfs --size=simlarge --input=../../data/bfs/graph1MW_6.txt --bm=rodinia

echo "==================Run Parsec (one csv per line)====================="
echo "-----blackscholes (1/7)-----"
python3 run_benchmark.py --prog=blackscholes --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=blackscholes --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=blackscholes --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=blackscholes --size=simnative --bm=parsec

echo "-----bodytrack (2/7)-----"
python3 run_benchmark.py --prog=bodytrack --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=bodytrack --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=bodytrack --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=bodytrack --size=simnative --bm=parsec

echo "-----facesim (3/7)-----"
python3 run_benchmark.py --prog=facesim --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=facesim --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=facesim --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=facesim --size=simnative --bm=parsec

echo "-----fluidanimate (4/7)-----"
python3 run_benchmark.py --prog=fluidanimate --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=fluidanimate --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=fluidanimate --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=fluidanimate --size=simnative --bm=parsec

echo "-----swaptions (5/7)-----"
python3 run_benchmark.py --prog=swaptions --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=swaptions --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=swaptions --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=swaptions --size=simnative --bm=parsec

echo "-----canneal (6/7)-----"
python3 run_benchmark.py --prog=canneal --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=canneal --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=canneal --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=canneal --size=simnative --bm=parsec

echo "-----streamcluster (7/7)-----"
python3 run_benchmark.py --prog=streamcluster --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=streamcluster --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=streamcluster --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=streamcluster --size=simnative --bm=parsec