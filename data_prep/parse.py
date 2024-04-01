import subprocess 
import os

wd = os.getcwd()
os.chdir("/home/yl3750/MultiorePerformancePrediction/benchmark/rodinia/openmp/bfs")
#subprocess.Popen("ls")

try: 
    ans = subprocess.check_output(["perf", "stat", "./bfs", "4", "../../data/bfs/graph1MW_6.txt"], text=True) 
    print(ans)   
except subprocess.CalledProcessError as e: 
    print(f"Command failed with return code {e.returncode}")

#ans = subprocess.call(["perf", "stat", "./bfs", "4", "../../data/bfs/graph1MW_6.txt"])

os.chdir(wd)

