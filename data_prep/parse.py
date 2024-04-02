import subprocess 
import os

wd = os.getcwd()
os.chdir("/home/yl3750/MultiorePerformancePrediction/benchmark/rodinia/openmp/bfs")
#subprocess.Popen("ls")

output_name = "bfs-t1"
metrics_lst = [#"cycles,instructions,cache-misses"
  # Hardware event
  "branch-instructions",
  "branch-misses",
  "cache-misses",
  "cache-references",
  "cpu-cycles",
  "instructions",
  "stalled-cycles-backend",
  "stalled-cycles-frontend",
  # Software event
  "alignment-faults",
  "bpf-output",
  "context-switches",
  "cpu-clock",
  "cpu-migrations",
  "dummy",
  "emulation-faults",
  "major-faults",
  "minor-faults",
  "page-faults",
  "task-clock",
  # Hardware cache event
  "L1-dcache-load-misses",
  "L1-dcache-loads",
  "L1-dcache-prefetch-misses",
  "L1-dcache-prefetches",
  "L1-icache-load-misses",
  "L1-icache-loads",
  "L1-icache-prefetches",
  "LLC-load-misses",
  "LLC-loads",
  "LLC-stores",
  "branch-load-misses",
  "branch-loads",
  "dTLB-load-misses",
  "dTLB-loads",
  "iTLB-load-misses",
  "iTLB-loads",
  # Kernel PMU event
  "branch-instructions",
  "branch-misses",
  "cache-misses",
  "cache-references",
  "cpu-cycles",
  "instructions",
  "msr/aperf/",
  "msr/mperf/",
  "msr/tsc/",
  "stalled-cycles-backend",
  "stalled-cycles-frontend",
  # Raw hardware event descriptor
  #rNNN
  #cpu/t1=v1[,t2=v2,t3 ...]/modifier (see 'man perf-list' on how to encode it)
  # Hardware breakpoint
  #mem:<addr>[/len][:access]                          
]
metrics = ','.join(metrics_lst)
#metrics = "cycles,instructions,cache-misses"

try:
    print("--------------------------------------") 
    #ans = subprocess.check_output(["perf", "stat","-o", f"{output_name}.csv", "--field-separator=,", "-e", metrics, "./bfs", "4", "../../data/bfs/graph1MW_6.txt"], text=True) 
    #ans = subprocess.run(["perf", "stat","-o", f"{output_name}.csv", "--field-separator=,", "-e", metrics, "./bfs", "4", "../../data/bfs/graph1MW_6.txt"], capture_output=True)
    ans = subprocess.run(["perf", "stat", "-e", metrics, "./bfs", "4", "../../data/bfs/graph1MW_6.txt"], capture_output=True)
    print("-----------------------------------------")
    #print(["perf", "stat","-o", f"{output_name}.csv", "--field-separator=,", "-e", metrics, "./bfs", "4", "../../data/bfs/graph1MW_6.txt"])
    print(ans)   
except subprocess.CalledProcessError as e: 
    print(f"Command failed with return code {e.returncode}")

#ans = subprocess.call(["perf", "stat", "./bfs", "4", "../../data/bfs/graph1MW_6.txt"])

os.chdir(wd)

