#!/usr/bin/env bash
# colab-guard sandbox tests. usage: t.sh <guard path>
GUARD=$1; G=$(cd "$(dirname "$0")/.." && pwd); export HOME=$G/home
export COLAB_GUARD_STOP_WAIT=0
C=$HOME/.config/colab-guard; F=$HOME/fake; pass=0; fail=0
reset(){ rm -rf "$C" "$F"; mkdir -p "$C" "$F"; : > "$F/active"; cp "$(dirname "$GUARD")/selfstop.py" "$C/selfstop.py" 2>/dev/null; }
ok(){ if eval "$2"; then pass=$((pass+1)); echo "  PASS $1"; else fail=$((fail+1)); echo "  FAIL $1"; fi; }
regline(){ echo "$1 $2 ${3:-}" >> "$C/sessions"; touch -d "${4:-2 hours ago}" "$C/born_$1"; }
past=$(( $(date +%s) - 100 )); future=$(( $(date +%s) + 7200 ))
g(){ bash "$GUARD" "$@"; }

echo "T1 two sessions past deadline, saver reads stdin"
reset; printf 'a1\nb2\n' > $F/active; regline a1 $past $G/tools/saver.sh; regline b2 $past $G/tools/saver.sh; g sweep
ok "both stopped" '[ ! -s $F/active ]'; ok "registry cleared" '[ ! -s $C/sessions ]'

echo "T2 idle twice -> save and stop"
reset; echo a1 > $F/active; regline a1 $future $G/tools/saver.sh; echo "GUARD_PROBE gpu=0 load5=0.10 armed=1" > $F/probe_out
g sweep; ok "still active after 1 idle sweep" 'grep -qx a1 $F/active'; ok "idle counter 1" '[ "$(cat $C/idle_a1)" = 1 ]'
g sweep; ok "stopped after 2 idle sweeps" '[ ! -s $F/active ]'; ok "saver ran" 'grep -q "saved a1" $C/guard.log'; ok "alert written" 'grep -q "idle for 2 sweeps" $C/alerts'; ok "no false unregistered alert for the session just stopped" '! grep -q "unregistered" $C/alerts'

echo "T3 busy resets idle counter"
reset; echo a1 > $F/active; regline a1 $future; echo "GUARD_PROBE gpu=0 load5=0.10 armed=1" > $F/probe_out; g sweep
echo "GUARD_PROBE gpu=97 load5=3.20 armed=1" > $F/probe_out; g sweep; ok "counter removed" '[ ! -f $C/idle_a1 ]'
echo "GUARD_PROBE gpu=0 load5=2.50 armed=1" > $F/probe_out; g sweep; ok "cpu-only work is busy" '[ ! -f $C/idle_a1 ]'
echo "GUARD_PROBE gpu=0 load5=1.02 armed=1" > $F/probe_out; g sweep; ok "one busy process (load 1.0) is busy" '[ ! -f $C/idle_a1 ]'
echo "GUARD_PROBE gpu=-1 load5=0.52 armed=1" > $F/probe_out; g sweep; ok "resting VM noise (0.52) counts as idle" '[ "$(cat $C/idle_a1)" = 1 ]'
ok "still active" 'grep -qx a1 $F/active'

echo "T4 no response 3 sweeps -> stop"
reset; echo a1 > $F/active; regline a1 $future; echo empty > $F/exec_mode
g sweep; g sweep; ok "active after 2" 'grep -qx a1 $F/active'; g sweep; ok "stopped after 3" '[ ! -s $F/active ]'
reset; echo a1 > $F/active; regline a1 $future; echo hang > $F/exec_mode
s=$(date +%s); COLAB_GUARD_PROBE_MAX=2 g sweep; e=$(( $(date +%s) - s )); ok "hung exec bounded (${e}s)" '[ $e -lt 15 ]'; ok "noresp 1" '[ "$(cat $C/noresp_a1)" = 1 ]'

echo "T5 grace: no probe right after registration"
reset; echo a1 > $F/active; g reg a1 2 >/dev/null; : > $F/calls; echo "GUARD_PROBE gpu=0 load5=0.0 armed=1" > $F/probe_out; g sweep
ok "no exec during grace" '! grep -q "^exec" $F/calls'; ok "born file exists" '[ -f $C/born_a1 ]'

echo "T6 re-arm when thread missing"
reset; echo a1 > $F/active; regline a1 $future; echo "GUARD_PROBE gpu=90 load5=2.0 armed=0" > $F/probe_out; g sweep
ok "arm logged" 'grep -q "arm a1: SELFSTOP armed" $C/guard.log'; ok "idle_min passed to selfstop" 'grep -q "IDLE_MIN = 90" $C/guard.log'

echo "T7 hold disables progress watch, deadline still applies"
reset; echo a1 > $F/active; regline a1 $future; echo "GUARD_PROBE gpu=0 load5=0.0 armed=1" > $F/probe_out; g hold a1 >/dev/null; : > $F/calls; g sweep; g sweep
ok "not stopped" 'grep -qx a1 $F/active'; ok "hold arms with idle 0" 'grep -q "IDLE_MIN = 0" $C/guard.log'
sed -i "s/ $future / $past /" $C/sessions; g sweep; ok "deadline still stops" '[ ! -s $F/active ]'

echo "T8 list failure alerts"
reset; echo a1 > $F/active; regline a1 $past; touch $F/listfail; g sweep; g sweep; ok "no alert yet" '[ ! -s $C/alerts ]'; g sweep; ok "alert on 3rd" 'grep -q "cannot list" $C/alerts'
rm $F/listfail; g sweep; ok "recovers and stops" '[ ! -s $F/active ]'

echo "T9 saver hangs -> still stopped"
reset; echo a1 > $F/active; printf '#!/bin/sh\nsleep 100\n' > $G/tools/hang.sh; regline a1 $past $G/tools/hang.sh
s=$(date +%s); COLAB_GUARD_SAVE_MAX=2 g sweep; e=$(( $(date +%s) - s )); ok "bounded (${e}s)" '[ $e -lt 15 ]'; ok "stopped" '[ ! -s $F/active ]'; ok "says SAVE FAILED" 'grep -q "SAVE FAILED" $C/alerts'

echo "T10 sweep killed mid-save earlier -> next sweep stops without saving again"
reset; echo a1 > $F/active; regline a1 $past $G/tools/hang.sh; touch $C/saved_a1
s=$(date +%s); g sweep; e=$(( $(date +%s) - s )); ok "no second save (${e}s)" '[ $e -lt 10 ]'; ok "stopped" '[ ! -s $F/active ]'

echo "T11 stop fails -> alert, stays registered, retried"
reset; echo a1 > $F/active; regline a1 $past; touch $F/stopfail; g sweep
ok "alert FAILED" 'grep -q "FAILED to stop a1" $C/alerts'; ok "still registered" 'grep -q "^a1 " $C/sessions'
rm $F/stopfail; g sweep; ok "retry stops" '[ ! -s $F/active ]'

echo "T12 reg validation"
reset
g reg "a.b" 2 2>/dev/null; ok "bad name refused" '! grep -q "a.b" $C/sessions'
g reg colab 2 2>/dev/null; ok "name colab refused" '! grep -q "^colab " $C/sessions'
g reg a1 abc 2>/dev/null; ok "bad hours refused" '! grep -q "^a1 " $C/sessions'
g reg a1 48 2>/dev/null; ok "48h refused" '! grep -q "^a1 " $C/sessions'
touch $C/budget_block_$(date +%Y-%m-%d); g reg a1 2 2>/dev/null; ok "budget block refuses new" '! grep -q "^a1 " $C/sessions'
rm $C/budget_block_*; g reg a1 2 >/dev/null; ok "normal reg ok" 'grep -q "^a1 " $C/sessions'
touch $C/budget_block_$(date +%Y-%m-%d); g reg a1 3 >/dev/null 2>&1; ok "extension allowed under block" '[ $(grep -c "^a1 " $C/sessions) = 1 ]'

echo "T13 bash-hook"
K="pkill"; FL="-f"; N="new"
h(){ jq -n --arg c "$1" --arg cwd "$G" '{tool_input:{command:$c}, cwd:$cwd}' | g bash-hook; }
hw(){ jq -n --arg c "$1" '{tool_input:{content:$c}}' | g bash-hook; }
den(){ echo "$1" | grep -q '"deny"'; }
ok "kill by pattern denied" 'den "$(h "$K $FL train.py")"'
ok "kill by pattern inside python string denied" 'den "$(h "colab exec -s x -c \"os.system(\\\"$K -9 $FL foo\\\")\"")"'
ok "pgrep allowed" '! den "$(h "pgrep -af train.py")"'
ok "stop without name denied" 'den "$(h "colab stop")"'
ok "stop after && denied" 'den "$(h "echo hi && timeout 60 colab stop")"'
ok "stop -s name allowed" '! den "$(h "colab stop -s e38")"'
ok "stop --help allowed" '! den "$(h "colab stop --help")"'
ok "new without reg denied" 'den "$(h "colab $N -s e38 --gpu L4")"'
ok "new with reg allowed" '! den "$(h "colab-guard reg e38 3 && colab $N -s e38 --gpu L4")"'
ok "mention in quotes allowed" '! den "$(h "echo \"use colab $N to open\"")"'
printf '#!/bin/bash\ntimeout 600 colab %s -s x --gpu L4\n' $N > $G/tools/bad_launch.sh
printf '#!/bin/bash\ncolab-guard reg x 3 || exit 1\ntimeout 600 colab %s -s x --gpu L4\n' $N > $G/tools/good_launch.sh
ok "script without reg denied" 'den "$(h "HOURS=3 bash tools/bad_launch.sh")"'
ok "script with reg allowed" '! den "$(h "HOURS=3 bash tools/good_launch.sh")"'
ok "Write of script without reg denied" 'den "$(hw "$(cat $G/tools/bad_launch.sh)")"'
ok "Write of script with reg allowed" '! den "$(hw "$(cat $G/tools/good_launch.sh)")"'
ok "Write with kill pattern denied" 'den "$(hw "x; $K $FL foo")"'

echo "T14 stop-hook"
reset; echo zz > $F/active
ok "unregistered blocks" 'echo "{}" | g stop-hook | grep -q "\"block\""'
ok "second time (hook active) only message" 'echo "{\"stop_hook_active\":true}" | g stop-hook | grep -q systemMessage'
regline zz $future; ok "registered -> message only" 'echo "{}" | g stop-hook | grep -q "active zz"'
: > $F/active; ok "no sessions -> silent" '[ -z "$(echo "{}" | g stop-hook)" ]'

echo "T15 registered but inactive -> one alert"
reset; regline a1 $future; g sweep; g sweep; ok "one lost alert" '[ "$(grep -c "not active" $C/alerts)" = 1 ]'

echo "T16 unregistered session is alerted, never stopped"
reset; echo mine > $F/active; g sweep; ok "alerted" 'grep -q "unregistered Colab session .mine." $C/alerts'; ok "not stopped" 'grep -qx mine $F/active'; ok "no stop call" '! grep -q "^stop" $F/calls'

echo "RESULT pass=$pass fail=$fail"
