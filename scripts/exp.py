import subprocess

ans = subprocess.run(['perf', 'list'], capture_output=True)

lst = []
for line in ans.stdout.decode().splitlines():
    line_lst = line.split()
    if line_lst:
        lst.append(line_lst[0])
print(lst)

