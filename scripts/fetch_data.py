# -*- coding: utf-8 -*-
"""Download the PlanetLab CPU-utilisation traces used in the study.

The traces are distributed by the CloudSim Plus project
(https://github.com/cloudsimplus/planetlab-workload-traces) and are not redistributed here.
They are unpacked to data/planetlab-workload-traces-master/, the default location read by
farda/data.py (override with the PLANETLAB_ROOT environment variable).
"""
import io
import os
import urllib.request
import zipfile

URL = "https://github.com/cloudsimplus/planetlab-workload-traces/archive/refs/heads/master.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data")


def main():
    target = os.path.join(DEST, "planetlab-workload-traces-master")
    if os.path.isdir(target):
        print("already present:", target)
        return
    os.makedirs(DEST, exist_ok=True)
    print("downloading", URL)
    with urllib.request.urlopen(URL, timeout=300) as r:
        data = r.read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(DEST)
    days = sorted(d for d in os.listdir(target) if d.startswith("2011"))
    print(f"{len(days)} trace days in {target}: {', '.join(days)}")


if __name__ == "__main__":
    main()
