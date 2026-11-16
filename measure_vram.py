"""
Measure peak GPU VRAM while running example_layered.py.

Usage:
  python measure_vram.py                  # benchmark 100³, 150³, 200³
  python measure_vram.py 80               # single size
  python measure_vram.py 100 200          # two sizes
  python measure_vram.py 100 --preconditioner jacobi
"""
import sys
import subprocess
import threading
import time
import math


def _idle_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()
    return int(out.split()[0])


def measure(size, preconditioner="multigrid"):
    Ne   = size ** 3
    Ndof = (size + 1) ** 3 * 3

    idle = _idle_mb()

    peak_mb = [idle]
    stop    = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL).strip()
                mb = int(out.split()[0])
                if mb > peak_mb[0]:
                    peak_mb[0] = mb
            except Exception:
                pass
            time.sleep(0.1)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    cmd = [sys.executable, "example_layered.py",
           "--size", str(size), "--vtk", "",
           "--preconditioner", preconditioner]
    t0 = time.perf_counter()
    subprocess.run(cmd, text=True)
    elapsed = time.perf_counter() - t0

    stop.set()
    poller.join()

    net = peak_mb[0] - idle
    return dict(size=size, Ne=Ne, Ndof=Ndof, idle_mb=idle,
                peak_mb=peak_mb[0], net_mb=net, elapsed=elapsed,
                preconditioner=preconditioner)


def print_result(r):
    Ne, Ndof, net = r["Ne"], r["Ndof"], r["net_mb"]
    print(f"\n{'='*64}")
    print(f"  Grid:            {r['size']}³  =  {Ne:,} voxels  |  {Ndof:,} DOFs")
    print(f"  Preconditioner:  {r['preconditioner']}")
    print(f"  Wall time:       {r['elapsed']:.1f} s")
    print(f"  Idle VRAM:       {r['idle_mb']} MB")
    print(f"  Peak VRAM:       {r['peak_mb']} MB  (net = {net} MB)")
    print(f"  Bytes / voxel:   {net * 1024**2 / Ne:.0f}")
    print(f"  Bytes / DOF:     {net * 1024**2 / Ndof:.0f}")
    print(f"{'='*64}")


def print_summary(results):
    print(f"\n{'='*72}")
    print(f"  {'Grid':<8} {'Voxels':>10} {'DOFs':>12} {'Net VRAM':>10}"
          f"  {'B/DOF':>6}  {'Time':>7}")
    print(f"  {'-'*68}")
    for r in results:
        bpd = r["net_mb"] * 1024**2 / r["Ndof"]
        print(f"  {r['size']}³{'':<5} {r['Ne']:>10,} {r['Ndof']:>12,}"
              f" {r['net_mb']:>8} MB  {bpd:>6.0f}  {r['elapsed']:>6.1f} s")

    if len(results) >= 2:
        # Linear fit: net_mb = A + B * Ne_millions
        xs = [r["Ne"] / 1e6 for r in results]
        ys = [r["net_mb"]   for r in results]
        n  = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        B = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / sum((x-mx)**2 for x in xs)
        A = my - B * mx

        print(f"\n  Linear fit:  VRAM ≈ {A:.0f} MB  +  {B:.1f} MB/M_voxel × Ne")

        # Extrapolate to 300M voxels and 48 GB
        vram_300 = A + B * 300
        avail    = 48 * 1024
        Ne_max   = (avail - A) / B if B > 0 else float("inf")
        N_max    = round(Ne_max * 1e6) ** (1/3) if Ne_max > 0 else 0

        print(f"  At 300M voxels (target):  {vram_300/1024:.1f} GB")
        print(f"  Ceiling at 48 GB:         {Ne_max:.0f} M voxels"
              f"  (≈ {int(round((Ne_max*1e6)**(1/3)))}³)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse sizes and optional --preconditioner flag
    sizes, prec = [], "multigrid"
    i = 0
    while i < len(args):
        if args[i] == "--preconditioner" and i + 1 < len(args):
            prec = args[i + 1]
            i += 2
        else:
            try:
                sizes.append(int(args[i]))
            except ValueError:
                pass
            i += 1

    if not sizes:
        sizes = [100, 150, 200]

    results = []
    for s in sizes:
        print(f"\nRunning {s}³ ({s**3:,} voxels) with {prec} …")
        r = measure(s, preconditioner=prec)
        print_result(r)
        results.append(r)

    if len(results) > 1:
        print_summary(results)
