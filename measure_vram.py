"""
Run example_layered.py and report peak VRAM and CuPy pool usage.
Usage: python measure_vram.py <size> [--preconditioner jacobi|multigrid]
"""
import sys
import subprocess
import threading
import time
import os

def monitor_and_run(size, preconditioner="jacobi", extra_args=None):
    Ne = size ** 3
    Ndof = (size + 1) ** 3 * 3

    peak_mb = [0]
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
                mb = int(out.split()[0])
                samples.append(mb)
                if mb > peak_mb[0]:
                    peak_mb[0] = mb
            except Exception:
                pass
            time.sleep(0.1)

    # idle baseline
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()
    idle_mb = int(out.split()[0])

    python = sys.executable
    cmd = [python, "example_layered.py",
           "--size", str(size),
           "--vtk", "",
           "--preconditioner", preconditioner]
    if extra_args:
        cmd += extra_args

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.perf_counter() - t0
    stop.set()
    poller.join()

    net_peak = peak_mb[0] - idle_mb
    print(f"\n{'='*64}")
    print(f"  Grid:  {size}³  =  {Ne:,} voxels  |  {Ndof:,} DOFs")
    print(f"  Preconditioner:  {preconditioner}")
    print(f"  Wall time:       {elapsed:.2f} s")
    print(f"  Idle VRAM:       {idle_mb} MB")
    print(f"  Peak VRAM:       {peak_mb[0]} MB  (net = {net_peak} MB)")
    print(f"  MB / voxel:      {net_peak / Ne * 1e6:.1f}  bytes/voxel")
    print(f"  MB / DOF:        {net_peak / Ndof * 1e6:.1f}  bytes/DOF")
    print(f"{'='*64}\n")
    return net_peak, Ne, Ndof

if __name__ == "__main__":
    args = sys.argv[1:]
    size = int(args[0]) if args else 100
    prec = "multigrid"
    for i, a in enumerate(args):
        if a == "--preconditioner" and i + 1 < len(args):
            prec = args[i + 1]
    monitor_and_run(size, preconditioner=prec)
