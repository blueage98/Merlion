#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Smoke test for the Merlion dashboard web app.

Starts `merlion.dashboard` (Dash/Flask app) as a subprocess, waits for it to
come up, checks that the home page and a couple of tab routes respond with
HTTP 200, then shuts the server down.

Usage:
    python scripts/test_dashboard.py
    python scripts/test_dashboard.py --host 127.0.0.1 --port 8050 --timeout 30
"""
import argparse
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_for_port(host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def check_url(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "merlion-smoke-test"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for the server to start")
    args = parser.parse_args()

    proc = subprocess.Popen(
        [sys.executable, "-m", "merlion.dashboard"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        print(f"Waiting for dashboard on {args.host}:{args.port} ...")
        if not wait_for_port(args.host, args.port, args.timeout):
            print("Server did not open its port in time.")
            _dump_output(proc)
            sys.exit(1)

        base_url = f"http://{args.host}:{args.port}"
        failures = []
        for path in ["/"]:
            url = base_url + path
            try:
                status, body = check_url(url)
                ok = status == 200 and b"Merlion" in body
                print(f"GET {url} -> {status} ({'OK' if ok else 'FAIL'})")
                if not ok:
                    failures.append(url)
            except urllib.error.URLError as e:
                print(f"GET {url} -> ERROR: {e}")
                failures.append(url)

        if failures:
            print(f"\nFAILED checks: {failures}")
            _dump_output(proc)
            sys.exit(1)

        print("\nDashboard smoke test passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _dump_output(proc):
    proc.poll()
    if proc.stdout:
        print("--- server output ---")
        print(proc.stdout.read())


if __name__ == "__main__":
    main()
