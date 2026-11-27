"""Minimal job queue: runs FE jobs (run_design.py argument lists) with a fixed
number of concurrent worker processes and records measured wall times."""
import os as _os
_ROOT = _os.environ.get('GYROID_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + '/'
import json, os, subprocess, sys, time

def main(jobfile, nworkers=2):
    jobs = json.load(open(jobfile))
    pending = [j for j in jobs if not os.path.exists(j['out'] + '_meta.json')]
    running = []
    logf = open(jobfile + '.log', 'a')
    while pending or running:
        while pending and len(running) < nworkers:
            j = pending.pop(0)
            cmd = [sys.executable, _ROOT + 'code/run_design.py'] + j['args'] + ['--out', j['out']]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            running.append((p, j, time.time()))
            print('start', j['out'], flush=True, file=logf)
        time.sleep(2)
        still = []
        for p, j, t0 in running:
            if p.poll() is None:
                still.append((p, j, t0))
            else:
                out = p.stdout.read()
                print('end %s rc=%d %.0fs %s' % (j['out'], p.returncode, time.time() - t0, out.strip()[-200:]), flush=True, file=logf)
        running = still
    print('ALL DONE', flush=True, file=logf)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
