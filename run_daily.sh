#!/bin/zsh
cd "$(dirname "$0")" || exit 1
sleep 60
.venv/bin/python src/gmail_label.py --apply >> run.log 2>&1
.venv/bin/python src/fix_german.py --apply >> run.log 2>&1
echo "done $(date)" >> run.log
