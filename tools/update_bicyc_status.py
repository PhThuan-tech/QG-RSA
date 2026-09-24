#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--status", required=True)
p.add_argument("--phase", required=True)
p.add_argument("--arm")
p.add_argument("--message")
p.add_argument("--exit-code", type=int)
p.add_argument("--decision")
a = p.parse_args()
path = Path(a.root) / "status.json"
previous = json.loads(path.read_text()) if path.exists() else {"history": []}
event = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "status": a.status,
    "phase": a.phase,
    "arm": a.arm,
    "message": a.message,
    "exit_code": a.exit_code,
    "decision": a.decision,
}
history = previous.get("history", []) + [event]
payload = {**event, "history": history}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
