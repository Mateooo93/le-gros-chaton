"""Persistent colab-mcp driver: keeps one server alive, connects the browser,
and verifies GPU. Run: python3 colab_driver.py --connect  (then it stays alive)
"""
import json, subprocess, sys, time, os, signal

VENV_BIN = "/tmp/colab-mcp-venv/bin/colab-mcp"
proc = subprocess.Popen([VENV_BIN], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, text=True,
                        stderr=subprocess.DEVNULL)

def rpc(method, params, mid, timeout=60):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params}) + "\n")
    proc.stdin.flush()
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("id") == mid:
            return d
    return {"error": "timeout"}

rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "driver", "version": "0.1"}}, 1)
time.sleep(3)

# List tools to see the stubs
r = rpc("tools/list", {}, 2)
tools = [t["name"] for t in r.get("result", {}).get("tools", [])]
print("TOOLS:", tools)

# Open the browser connection
print("Opening Colab browser connection...")
r = rpc("tools/call", {"name": "open_colab_browser_connection", "arguments": {}}, 3)
text = r.get("result", {}).get("content", [{}])[0].get("text", str(r))
print("CONNECT:", text[:300])

if "not connected" in text.lower() or "Connection successful" not in text:
    print("Waiting for browser connect... (make sure the Colab tab is open and connected)")
    for i in range(30):
        time.sleep(5)
        r = rpc("tools/call", {"name": "open_colab_browser_connection", "arguments": {}}, 3)
        text = r.get("result", {}).get("content", [{}])[0].get("text", "")
        if "Connection successful" in text:
            print("CONNECTED:", text[:200])
            break

# Verify GPU
print("Checking GPU...")
r = rpc("tools/call", {"name": "add_code_cell", "arguments": {
    "code": "import torch\nprint('GPU:', torch.cuda.is_available())\nif torch.cuda.is_available(): print(torch.cuda.get_device_name(0))\nprint('torch', torch.__version__)"
}}, 4)
print("ADD CELL:", json.dumps(r)[:200])
time.sleep(2)
r = rpc("tools/call", {"name": "run_code_cell", "arguments": {}}, 5, timeout=120)
print("RUN CELL:", json.dumps(r)[:500])

proc.terminate()
