---
name: long-running-jobs
description: Starts commands expected to run for more than a few minutes - dataset builds, training runs - detached from the Claude Code session, so they survive it ending and leave a log that shows whether they finished or died. Use before launching any long job, when checking whether one is still running or what happened to it, and before restarting an interrupted one, where rerunning from the start silently duplicates its output.
---

# Long-running jobs

The Bash tool's `run_in_background` keeps a command alive across turns but not across sessions. When the
Claude Code session ends, the job dies with it - no error, and possibly before writing anything. A
2.5-hour dataset build was lost that way. Detach anything that should outlast a few minutes.

## Start

```bash
.claude/skills/long-running-jobs/scripts/run_detached.sh <log-file> '<command>'
```

The command runs under `bash -c`, so pipelines work. It gets its own session (`setsid`) and ignores the
hangup (`nohup`), then appends to the log between `=== start ... session N ===` and `=== end ... exit N ===`.
The script prints the start line; its session id is the handle for everything below.

Two things keep the log honest. Without them it stays empty for long stretches and the job looks hung:

- `python -u`: Python block-buffers stdout whenever it is not writing to a terminal.
- `grep --line-buffered` in any pipeline: grep otherwise holds output back in 4 KB blocks.

## Confirm it is working, not just alive

A process can exist without doing anything useful. Wait for the job's first meaningful line, and for a
job that produces data, check the first real piece of output before trusting the rest:

```bash
timeout 120 bash -c 'until grep -q "<first expected line>" <log-file>; do sleep 2; done'
```

## Check on it later

```bash
ps -o pid,etime,cmd --sid <session>            # no rows: it is not running
grep -E '^=== (start|end)' <log-file> | tail -2
```

| running? | end marker | meaning |
|---|---|---|
| yes | none | still going |
| no | none | killed together with its wrapper - reboot, or `pkill -s` |
| no | `exit 0` | finished |
| no | `exit 137` / `143` | the job alone was killed: SIGKILL (often out of memory) / SIGTERM |
| no | other non-zero | the job failed; the lines above the marker say why |

A notification that a background task "didn't finish before the previous session ended" means a
session-bound job was killed. Check its outputs rather than assuming it completed.

## Stop it

```bash
pkill -s <session>                             # the job and everything it started
```

Avoid `pkill -f <pattern>`. It matches full command lines, including that of the shell running the
`pkill` whenever the pattern appears in its own command - a heredoc, a `grep`, an `echo` - and it has
killed the command doing the killing. If a pattern is unavoidable, run `pgrep -af '<pattern>'` first to
see exactly what it would hit.

## Resume it

A job that writes output as it goes, restarted from the beginning, writes all of it again. Before
relaunching:

1. Establish that it died (above) and what it already produced, using read-only commands - count its
   files, shards or rows.
2. Resume past that point with the job's own mechanism, preferring one that seeks over one that
   re-reads. The dataset builder (`tools/build_wake_vision.py`, step 04) shows the difference:
   `--skip-rows` streams every row it skips, 2.5 hours for the first 268k, while `--start-file` jumps
   to a parquet shard at no cost.
3. Check that the first new output does not overlap the old; for data, hash it against what exists. A
   probe passing beforehand is not the same as the real run being right.

Training (`training/train.py`, step 04) has no resume: a killed run starts again from epoch 1.
