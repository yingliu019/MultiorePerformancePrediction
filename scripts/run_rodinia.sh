# nohup ./data_gen.sh 2>&1 > output.txt &
# tail -f output.txt

python3 run_rodinia.py --prog=bfs --size=S --input=../../data/bfs/graph4096.txt
python3 run_rodinia.py --prog=bfs --size=M --input=../../data/bfs/graph65536.txt
python3 run_rodinia.py --prog=bfs --size=L --input=../../data/bfs/graph1MW_6.txt