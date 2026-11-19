#!/usr/bin/env python3
"""
Send the evening run's queued Telegram messages as one brief (.github/workflows/engine.yml, last step before the
commit). Every script in the run queues to the file named by TELEGRAM_BATCH instead of sending on its own; this
reads that file, orders it (utils/telegram_notifier.BATCH_ORDER) and sends it.

    python scripts/send_brief.py /tmp/telegram_batch.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv) -> int:
    path = argv[1] if len(argv) > 1 else os.environ.get("TELEGRAM_BATCH_FILE", "/tmp/telegram_batch.jsonl")
    os.environ.pop("TELEGRAM_BATCH", None)                  # send for real from here
    from utils.telegram_notifier import send_batch
    ok, info = send_batch(path)
    print(f"[brief] {'sent' if ok else 'FAILED'}: {info}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
