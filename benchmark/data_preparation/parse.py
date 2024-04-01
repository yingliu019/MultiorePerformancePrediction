import subprocess 
try: 
    ans = subprocess.check_output(["python", "--version"], text=True) 
    print(type(ans), ans) 
except subprocess.CalledProcessError as e: 
    print(f"Command failed with return code {e.returncode}")
