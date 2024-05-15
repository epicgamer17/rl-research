import time
import subprocess
from subprocess import Popen

try:
    cmd = f"./bin/test_py"
    print("running cmd:", cmd)
    go_proc = Popen(cmd.split(" "), stdin=subprocess.PIPE, text=True)
    time.sleep(5)
    print("Training complete")
except Exception as e:
    print(e)
finally:
    stdout, stderr = go_proc.communicate("\n\n")
    print(f"stdout:{stdout}")
    print(f"stderr:{stderr}")

print("done")
