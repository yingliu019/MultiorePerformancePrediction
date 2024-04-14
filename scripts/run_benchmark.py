import datetime
import getopt
import os
import re
import subprocess 
import sys
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
    # 'node-load-misses',
    # 'node-loads',
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

def parse_perf_list():
    ans = subprocess.run(['perf', 'list'], capture_output=True)
    
    lst = []
    for line in ans.stdout.decode().splitlines():
        line_lst = line.split()
        if line_lst:
            metrics = line_lst[0]
            if metrics.startswith('sdt_'):
                continue
            elif metrics == 'rNNN':
                continue
            elif metrics.startswith('cpu/t1'):
                continue
            elif metrics.startswith('mem:<addr>'):
                continue
            lst.append(line_lst[0])
    return lst

def get_rodinia_command(prog_name, threads, input_file) -> list:
    if prog_name == 'bfs':
        return ['./bfs', f'{threads}', input_file]
    elif prog_name == 'kmeans':
        return ['./kmeans_openmp/kmeans', '-n', f'{threads}', '-i', input_file]
    elif prog_name == 'lavaMD':
        return ['./lavaMD', '-cores', f'{threads}', '-boxes1d', input_file]
    elif prog_name == 'myocyte':
        return ['./myocyte.out', input_file, '1', '0', f'{threads}']
    else:
        return []

def get_parsec_command(prog_name, threads, size):
    return ['parsecmgmt', '-a', 'run', '-p', prog_name, '-c', 'gcc', '-i', size, '-n', f'{threads}']

def get_rodinia_time(ans, prog) -> float:
    if prog == 'bfs':    
        pattern = r'Compute time: ([\d.]+)'
        match = re.search(pattern, ans.stdout.decode())

    # Extract the compute time if found
    if match:
        compute_time = float(match.group(1))
        # print(f'Compute time: {compute_time:.6f}')
        return compute_time
    else:
        return None
        
def get_real_time(ans):
    pattern = r"real\t0m(\d+\.\d+)s"
    # Search for the pattern in the text
    match = re.search(pattern, ans.stdout.decode())
    #print(ans, match)
    if match:
      # Extract the captured time in seconds (float)
      real_time = float(match.group(1))
      return real_time
    return None

def parse_prog_metrics(compute_time, output_path) -> dict:
    time.sleep(3)
    df = pd.read_csv(output_path, skiprows=1, header=None)
    prod_dict = {'compute_time': compute_time}
    for _, row in df.iterrows():
        if list(row)[2]:
            prod_dict[list(row)[2]] = list(row)[0]
    return prod_dict

def parse_parsec_time(data):
    count = total = 0
    pattern = r"real\s+(.+)s"
    for line in data.stdout.decode().splitlines():
        match = re.search(pattern, line)
        #print('@@@@@@@@@@@@@', line)
        if match:
            real_time = match.group(1)
            # Convert the time string to seconds (assuming format "mm.sss")
            minutes, seconds = real_time.split("m")
            total += float(minutes) * 60 + float(seconds)
            count += 1
            print(total, count)
    return total / count

def parse_rodinia_time(data, prog):
    count = total = 0
    found = False
    pattern = r'unable to match'
    if prog == 'bfs':
        pattern = r'Compute time: ([\d.]+)' 
    elif prog == 'kmeans':
        pattern = r'Time for process: ([\d.]+)'
    for line in data.stdout.decode().splitlines():
        match = re.search(pattern, line)
        #print('@@@@@@@@@@@@@', line)
        if match:
            real_time = float(match.group(1))
            total += real_time
            count += 1
            #print(total, count)
        if prog in ['lavaMD', 'myocyte']:
            if found:
                real_time = float(line.split()[0])
                total += real_time
                count += 1
                found = False
                #print(total, count)
            elif 'Total time' in line:
                found = True
    return total / count

def get_program_metrics(run_id, prog_name, threads, input_file, benchmark, size, perf_list, repeat=5) -> dict:
    if benchmark == 'rodinia':
        run_command_lst = get_rodinia_command(prog_name, threads, input_file)
    elif benchmark == 'parsec':
        run_command_lst = get_parsec_command(prog_name, threads, size)

    output_path = f'/home/yl3750/MultiorePerformancePrediction/data/prog_benchmark/{run_id}.csv'
    perf_str = ','.join(perf_list)
    call_list = [
            'perf', 'stat', 
            '-o', output_path, '--field-separator=,',
            '-r', f'{repeat}',
            '-e', perf_str,
            ] + run_command_lst
    #print(' '.join(call_list))

    try:
        ans = subprocess.run([#'time',
            'perf', 'stat', 
            '-o', output_path, '--field-separator=,',
            '-r', f'{repeat}',
            '-e', perf_str, #METRICS,
            ] + run_command_lst, capture_output=True)
    except subprocess.CalledProcessError as e: 
        print(f'Command failed with return code {e.returncode}')
    #print(ans)
    #parse_parsec_time(ans)
    if benchmark == 'rodinia':
        compute_time = parse_rodinia_time(ans, prog_name)
    elif benchmark == 'parsec':
        compute_time = parse_parsec_time(ans)

    return parse_prog_metrics(compute_time, output_path)


def get_host_status():
    host_status = subprocess.run(['sar', '-u', '-r', '1', '3'], capture_output=True)
    return host_status.stdout.decode()


# def get_host_name():
#     ans = subprocess.run(["hostname"], capture_output=True)
#     return ans.stdout.decode().split('.')[0]


def parse_host_util_and_speedup(run_id, prog, threads, host_status, speed_up):
    lines = host_status.splitlines()
    for index, line in enumerate(lines):
        #if line.startswith('Linux'):
        #    hostname = line.split()[2].lstrip('(').rstrip(')')   
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
    return {'run_id': run_id, 'program': prog, 'threads': threads, #'hostname': hostname, 
            'host_cpu_user':average_cpu_user , 'host_cpu_system': average_cpu_system, 
            'host_cpu_idle': average_cpu_idle, 'host_memused': average_memused, 'speed_up': speed_up}

def get_host_spec():
    data_dict = {}
    ans = subprocess.run(["hostname"], capture_output=True)
    hostname = ans.stdout.decode().split('.')[0]
    data_dict['hostname'] = hostname

    host_spec1 = subprocess.run(['lscpu'], capture_output=True)
    data1 = host_spec1.stdout.decode()

    for line in data1.splitlines():
      key, val = line.replace(u'\xa0', u' ').split(':')
      if key == 'Flags':
          continue
      data_dict[key] = val.strip()
    
    host_spec2 = subprocess.run(['inxi', '-Sm'], capture_output=True)
    #print(host_spec2)
    data2 = host_spec2.stdout.decode().replace(u'\x0312', u' ').replace(u'\x03', u':')
    #print(data2)
    # Regex patterns to extract the required information
    #host_pattern = r"Host:\s*([^\s]+)"
    #kernel_pattern = r"Kernel:\s*([^\s]+)"
    #arch_pattern = r"arch:\s*([^\s]+)"
    memory_total_pattern = r"total:\s*([^\s]+ [^\s]+)"
    #memory_available_pattern = r"available:\s*([^\s]+ [^\s]+)"
    #memory_used_pattern = r"used:\s*([^\s]+ [^\s]+)"
    
    # Using search to find the first occurrence and extract data
    #hostname = re.search(host_pattern, data2).group(1).split('.')[0]
    #kernel_version = re.search(kernel_pattern, data).group(1)
    #architecture = re.search(arch_pattern, data).group(1)
    memory_total = re.search(memory_total_pattern, data2).group(1)
    #memory_available = re.search(memory_available_pattern, data).group(1)
    #memory_used = re.search(memory_used_pattern, data).group(1)
    
    # Print extracted information
    data_dict['total_memory'] = memory_total

    #print(data_dict)
    return data_dict

def run_openmp(prog, input_file, size, benchmark):
    data = {}
    
    if benchmark == 'rodinia':
        child = os.path.join(OPENMP_PROG_DIR, prog)
        os.chdir(child)

    base_time = float('inf')
    #hostname = get_host_name()
    host_spec = get_host_spec()
    hostname = host_spec['hostname']
    speed_up = 0
    perf_list = parse_perf_list()
    for index, threads in enumerate([1, 1, 2, 4, 8, 16, 32, 64, 128]):
        if prog == 'streamcluster' and threads == 128:
            continue
        run_id = f'{prog}_{size}_t{threads}_{hostname}'
        current_time = datetime.datetime.now()
        print(f'running {index}: {run_id} at {current_time} ...')
        #program_metrics = get_program_metrics(run_id, prog, threads, input_file, benchmark, size, perf_list)
        if index == 0:
            program_metrics = get_program_metrics(run_id, prog, threads, input_file, benchmark, size, perf_list, repeat=2)
            print('finishing cold start')
            continue        
        program_metrics = get_program_metrics(run_id, prog, threads, input_file, benchmark, size, perf_list)        
        host_status = get_host_status()        
        program_metrics['size'] = size
        program_metrics['run_time'] = current_time
        program_metrics['benchmark'] = benchmark
        if threads == 1:
            base_time = program_metrics['compute_time']
        speed_up = base_time / program_metrics['compute_time']
        host_util_and_speedup = parse_host_util_and_speedup(run_id, prog, threads, host_status, speed_up)
        data[run_id] = program_metrics | host_util_and_speedup | host_spec
        time.sleep(10)

    df = pd.DataFrame.from_dict(data, orient='index')
    time_str = current_time.strftime("%Y%m%d%H%M")
    df.to_csv(f'/home/yl3750/MultiorePerformancePrediction/data/training_data/{prog}_{size}_{hostname}_{time_str}.csv', index=False)
    print(f'Finished saving {prog}_{size}_{hostname} result.')

if __name__ == '__main__':
    wd = os.getcwd()
    input_file = ''
    try:
        arguments, values = getopt.getopt(sys.argv[1:], "psib", ["prog=", "size=", "input=", "bm="])
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-p", "--prog"):
                prog = currentValue
            elif currentArgument in ("-s", "--size"):
                size = currentValue
            elif currentArgument in ("-i", "--input"):
                input_file = currentValue  
            elif currentArgument in ("-b", "--bm"):
                benchmark = currentValue        
    except getopt.error as err:
        print(str(err))

    run_openmp(prog, input_file, size, benchmark)

    os.chdir(wd)
