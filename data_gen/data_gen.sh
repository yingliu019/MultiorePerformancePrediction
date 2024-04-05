# nohup ./data_gen.sh 2>&1 > output.txt &
# tail -f output.txt

python3 run_rodinia.py --prog=bfs --size=U --input=../../data/bfs/graph1MW_6.txt
