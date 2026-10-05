#!/bin/zsh
cd "$(dirname "$0")" || exit 1
sleep 60
.venv/bin/python src/step8_launch.py --apply >> run.log 2>&1
.venv/bin/python src/step8_monitor.py >> run.log 2>&1
.venv/bin/python src/step8_retrain.py >> run.log 2>&1
echo "done $(date)" >> run.log
