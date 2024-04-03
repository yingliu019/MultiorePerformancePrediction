import subprocess 
import os
import re

metrics_lst = [
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
  # rNNN
  # cpu/t1=v1[,t2=v2,t3 ...]/modifier (see 'man perf-list' on how to encode it)
  # Hardware breakpoint
  # mem:<addr>[/len][:access]                          
]
metrics = ','.join(metrics_lst)

def get_compute_time(ans) -> float:
    print("------------", type(ans))
    print(ans)
    pattern = r"Compute time: ([\d.]+)"
    match = re.search(pattern, ans.stdout.decode())

    # Extract the compute time if found
    if match:
        compute_time = float(match.group(1))
        print(f"Compute time: {compute_time:.6f}")
        return compute_time
    else:
        #print("Compute time not found in process output")
        return None

def get_program_metrics(thread_count=1, repeat = 2) -> float:
    prog_name = "bfs"
    try:
        ans = subprocess.run(["perf", "stat", "-o", f"/home/yl3750/MultiorePerformancePrediction/data/prog_benchmark/{prog_name}_t{thread_count}.csv", "--field-separator=,",
                            "-r", f"{repeat}",
                            "-e", metrics, 
                            "./bfs", f"{thread_count}", "../../data/bfs/graph1MW_6.txt"], capture_output=True)
    except subprocess.CalledProcessError as e: 
        print(f"Command failed with return code {e.returncode}")
    
    res = get_compute_time(ans)
    return res

def get_host_name():
    # add utilization
    return "cruntcy1"

def iterate_folder(parent_folder, host_spec):
    
    for child in parent_folder:
        child = "/home/yl3750/MultiorePerformancePrediction/benchmark/rodinia/openmp/bfs"
        os.chdir(child)
        # baseline
        prog_metrics = get_program_metrics()
        # baseline_time = prog_metrics['time']
        # merge {run id, prog metrics, host spec, thread count, runtime, speed up}

        # experiment
        # for thread in [2,4,8]:
        #     prog_metrics = get_program_metrics(thread_count=thread)
        #     run_time = prog_metrics['time']    
        #     merge {run id, prog metrics, host spec, thread count, runtime, speed up}   
        break


if __name__ == "__main__":
    wd = os.getcwd()
    host_spec = get_host_name()
    parent_folder = "1"
    iterate_folder(parent_folder, host_spec)
    os.chdir(wd)

