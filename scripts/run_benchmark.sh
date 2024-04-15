# cd /home/yl3750/MultiorePerformancePrediction/scripts
# nohup ./run_benchmark.sh 2>&1 > run_benchmark.txt &
# tail -f run_benchmark.txt

: <<'COMMENT'
echo "==================Preparation (one time per host)====================="
module load python-3.12
module load gcc-12.2

cd /home/yl3750/MultiorePerformancePrediction/benchmark/parsec-benchmark
chmod +x env.sh
source env.sh
COMMENT

echo "==================Run Parsec (one csv per line)====================="
#: <<'COMMENT'
echo "-----blackscholes (1/8)-----"
python3 run_benchmark.py --prog=blackscholes --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=blackscholes --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=blackscholes --size=simlarge --bm=parsec
#python3 run_benchmark.py --prog=blackscholes --size=simnative --bm=parsec

echo "-----bodytrack (2/8)-----"
python3 run_benchmark.py --prog=bodytrack --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=bodytrack --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=bodytrack --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=bodytrack --size=simnative --bm=parsec

echo "-----facesim (3/8)-----"
python3 run_benchmark.py --prog=facesim --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=facesim --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=facesim --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=facesim --size=simnative --bm=parsec

echo "-----fluidanimate (4/8)-----"
python3 run_benchmark.py --prog=fluidanimate --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=fluidanimate --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=fluidanimate --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=fluidanimate --size=simnative --bm=parsec

echo "-----swaptions (5/8)-----"
python3 run_benchmark.py --prog=swaptions --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=swaptions --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=swaptions --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=swaptions --size=simnative --bm=parsec

echo "-----canneal (6/8)-----"
python3 run_benchmark.py --prog=canneal --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=canneal --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=canneal --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=canneal --size=simnative --bm=parsec

echo "-----streamcluster (7/8)-----"
python3 run_benchmark.py --prog=streamcluster --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=streamcluster --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=streamcluster --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=streamcluster --size=simnative --bm=parsec
#COMMENT

echo "-----freqmine (8/8)-----"
python3 run_benchmark.py --prog=freqmine --size=simsmall --bm=parsec
python3 run_benchmark.py --prog=freqmine --size=simmedium --bm=parsec
python3 run_benchmark.py --prog=freqmine --size=simlarge --bm=parsec
# python3 run_benchmark.py --prog=freqmine --size=simnative --bm=parsec

echo "==================Run Rodinia (one csv per line)====================="
echo "-----bfs (1/3)-----"
python3 run_benchmark.py --prog=bfs --size=simsmall --input=../../data/bfs/graph4096.txt --bm=rodinia
python3 run_benchmark.py --prog=bfs --size=simmedium --input=../../data/bfs/graph65536.txt --bm=rodinia
python3 run_benchmark.py --prog=bfs --size=simlarge --input=../../data/bfs/graph1MW_6.txt --bm=rodinia

echo "-----kmeans (2/3)-----"
python3 run_benchmark.py --prog=kmeans --size=U --input=../../data/kmeans/kdd_cup --bm=rodinia

echo "-----lavaMD (3/3)-----"
python3 run_benchmark.py --prog=lavaMD --size=simsmall --input=10 --bm=rodinia
#python3 run_benchmark.py --prog=lavaMD --size=simmedium --input=100 --bm=rodinia
#python3 run_benchmark.py --prog=lavaMD --size=simlarge --input=1000 --bm=rodinia

echo "-----myocyte(4/4)-----"
python3 run_benchmark.py --prog=myocyte --size=simsmall --input=100 --bm=rodinia
#python3 run_benchmark.py --prog=heartwall --size=simmedium --input=200 --bm=rodinia
#python3 run_benchmark.py --prog=heartwall --size=simlarge --input=2000 --bm=rodinia

#COMMENT

echo "==================Script Ended.====================="

