import os
import re
import subprocess 
import time

import pandas as pd


# Constants.
METRICS_LIST = [
    # Hardware event
    'branch-instructions',
    'branch-misses',
    'cache-misses',
    'cache-references',
    'cpu-cycles',
    'instructions',
    'stalled-cycles-backend',
    'stalled-cycles-frontend',
    # Software event
    'alignment-faults',
    'bpf-output',
    'context-switches',
    'cpu-clock',
    'cpu-migrations',
    'dummy',
    'emulation-faults',
    'major-faults',
    'minor-faults',
    'page-faults',
    'task-clock',
    # Hardware cache event
    'L1-dcache-load-misses',
    'L1-dcache-loads',
    'L1-dcache-prefetch-misses',
    'L1-dcache-prefetches',
    'L1-icache-load-misses',
    'L1-icache-loads',
    'L1-icache-prefetches',
    'LLC-load-misses',
    'LLC-loads',
    'LLC-stores',
    'branch-load-misses',
    'branch-loads',
    'dTLB-load-misses',
    'dTLB-loads',
    'iTLB-load-misses',
    'iTLB-loads',
    # Kernel PMU event
    'branch-instructions',
    'branch-misses',
    'cache-misses',
    'cache-references',
    'cpu-cycles',
    'instructions',
    'msr/aperf/',
    'msr/mperf/',
    'msr/tsc/',
    'stalled-cycles-backend',
    'stalled-cycles-frontend',
    # Raw hardware event descriptor
    # rNNN
    # cpu/t1=v1[,t2=v2,t3 ...]/modifier (see 'man perf-list' on how to encode it)
    # Hardware breakpoint
    # mem:<addr>[/len][:access]                          
]
METRICS = ','.join(METRICS_LIST)

OPENMP_PROG_DIR ='/home/yl3750/MultiorePerformancePrediction/benchmark/rodinia/openmp'
PROGS = [
    #'b+tree', 
    #'backprop', 
    'bfs', 
    #  'cfd', 
    #  'heartwall', 
    #  'hotspot', 
    #  'hotspot3D', 
    #  'kmeans', 
    #  'lavaMD', 
    #  'leukocyte', 
    #  'lud', 
    #  'mummergpu', 
    #  'myocyte', 
    #  'nn', 
    #  'nw', 
    #  'particlefilter', 
    #  'pathfinder', 
    #  'srad', 
    #  'streamcluster'
]


def get_run_command(prog_name, threads) -> list:
    if prog_name == 'bfs':
        return ['./bfs', f'{threads}', '../../data/bfs/graph1MW_6.txt']
    else:
        return []


def get_compute_time(ans) -> float:
    pattern = r'Compute time: ([\d.]+)'
    match = re.search(pattern, ans.stdout.decode())

    # Extract the compute time if found
    if match:
        compute_time = float(match.group(1))
        # print(f'Compute time: {compute_time:.6f}')
        return compute_time
    else:
        # print('Compute time not found in process output')
        return None

def parse_prog_metrics(compute_time, output_path) -> dict:
    df = pd.read_csv(output_path, skiprows=1, header=None)
    prod_dict = {'compute_time': compute_time}
    for index, row in df.iterrows():
        if list(row)[2]:
            prod_dict[list(row)[2]] = list(row)[0]
    return prod_dict

def get_program_metrics(run_id, prog_name, threads, repeat=5) -> dict:
    run_command_lst = get_run_command(prog_name, threads)
    output_path = f'/home/yl3750/MultiorePerformancePrediction/data/prog_benchmark/{run_id}_{prog_name}_t{threads}.csv'
    try:
        ans = subprocess.run([
            'perf', 'stat', 
            '-o', output_path, '--field-separator=,',
            '-r', f'{repeat}',
            '-e', METRICS,
            ] + run_command_lst, capture_output=True)
    except subprocess.CalledProcessError as e: 
        print(f'Command failed with return code {e.returncode}')
    
    compute_time = get_compute_time(ans)
    return parse_prog_metrics(compute_time, output_path)


def get_host_status():
    host_status = subprocess.run(['sar', '-u', '-r', '1', '3'], capture_output=True)
    return host_status.stdout.decode()


def parse_host_spec_and_speedup(run_id, prog, threads, host_status, speed_up):
    lines = host_status.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('Linux'):
            hostname = line.split()[2].lstrip('(').rstrip(')')   
        if r'%idle' in line and 'Average:' in line and 'CPU' in line:
            fields = line.split()
            idle_index = fields.index(r'%user')
            average_cpu_user = float(lines[index+1].split()[idle_index])
            idle_index = fields.index(r'%system')
            average_cpu_system = float(lines[index+1].split()[idle_index])
            idle_index = fields.index(r'%idle')
            average_cpu_idle = float(lines[index+1].split()[idle_index])
        if r'%memused' in line:
            fields = line.split()
            idle_index = fields.index(r'%memused')
            average_memused = float(lines[index+1].split()[idle_index])
    return {'run_id': run_id, 'program': prog, 'threads': threads, 'hostname': hostname, 
            'host_cpu_user':average_cpu_user , 'host_cpu_system': average_cpu_system, 
            'host_cpu_idle': average_cpu_idle, 'host_memused': average_memused, 'speed_up': speed_up}


def run_openmp():
    run_id = 0
    data = {}
    for prog in PROGS:
        child = os.path.join(OPENMP_PROG_DIR, prog)
        os.chdir(child)

        base_time = float('inf')
        for threads in [1, 2, 4, 8, 16]:
            print(f'running {prog} with {threads} threads...', )
            host_status = get_host_status()
            program_metrics = get_program_metrics(run_id, prog, threads)
            if threads == 1:
                base_time = program_metrics['compute_time']
            speed_up = base_time / program_metrics['compute_time']
            host_spec_and_speedup = parse_host_spec_and_speedup(run_id, prog, threads, host_status, speed_up)
            data[run_id] = program_metrics | host_spec_and_speedup
            time.sleep(10)
            run_id += 1
    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_csv('/home/yl3750/MultiorePerformancePrediction/data/training.csv', index=False)


if __name__ == '__main__':
    wd = os.getcwd()
    run_openmp()
    os.chdir(wd)
