#!/usr/bin/env bash
# Run a shell command detached from the Claude Code session, so it outlives it.
#
#   run_detached.sh <log-file> '<command>'
#
# The command runs under bash -c, so pipelines work. Output is appended to the
# log between a start marker and an end marker. The end marker carries the exit
# code; a killed job never writes one, which is how a finished job is told apart
# from a dead one afterwards.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $(basename "$0") <log-file> '<command>'" >&2
    exit 2
fi
log=$1
cmd=$2
mkdir -p "$(dirname "$log")"
before=$(grep -c '^=== start' "$log" 2>/dev/null || true)

# setsid: its own session, outside the Claude Code process group.
# nohup:  ignore the hangup sent when the session that started it ends.
# stdin from /dev/null: there will be no terminal left to read from.
# Inside, $$ is the session leader, so it doubles as the session id.
setsid nohup bash -c '
    log=$1; cmd=$2
    echo "=== start $(date -Iseconds) session $$ ===" >> "$log"
    echo "    $cmd" >> "$log"
    bash -c "$cmd" >> "$log" 2>&1
    status=$?
    echo "=== end $(date -Iseconds) exit $status ===" >> "$log"
' _ "$log" "$cmd" > /dev/null 2>&1 < /dev/null &
disown

# Wait for this launch's start marker and print it: its session id is the
# handle for checking on and stopping the job.
started=0
for _ in $(seq 40); do
    now=$(grep -c '^=== start' "$log" 2>/dev/null || true)
    if [ "${now:-0}" -gt "${before:-0}" ]; then started=1; break; fi
    sleep 0.25
done
if [ "$started" -ne 1 ]; then
    echo "no new start marker in $log after 10 s" >&2
    exit 1
fi
grep '^=== start' "$log" | tail -1
