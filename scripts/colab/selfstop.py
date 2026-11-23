# Colab 쪽 이중 안전장치 (colab-guard arm이 부른다). kernel 안 thread가 1분마다 확인하고 runtime을 스스로 반납한다.
# 로컬 컴퓨터가 꺼져 있어도 세션이 꺼진다. 반납 조건은 둘이다.
#   1) /content/selfstop_deadline(epoch 초)이 지났다.
#   2) IDLE_MIN분 동안 계속 쉬었다(GPU 사용률 5% 미만이고 5분 load 평균 IDLE_LOAD 미만). 0이면 이 조건을 끈다.
#      IDLE_LOAD 0.8은 11/16 실측으로 정했다: 쉬는 VM은 0.35~0.5, 프로세스 하나만 일해도 1.0 이상이다.
#      pipeline이 시작되지 않았거나 일찍 죽은 세션이 기한까지 CU를 쓰는 것을 막는다(E36).
# 다시 부르면 기한과 IDLE_MIN 파일만 바꾼다(연장). thread는 하나만 띄운다.
# 주의: kernel이 재시작되면 thread가 사라진다. colab-guard sweep이 확인해서 다시 건다.
SECONDS = 4 * 3600
IDLE_MIN = 90
IDLE_LOAD = 0.8
import subprocess, threading, time
from google.colab import runtime

DL = '/content/selfstop_deadline'
IDLE = '/content/selfstop_idle_min'
LOGF = '/content/selfstop.log'
open(DL, 'w').write(str(int(time.time() + SECONDS)))
open(IDLE, 'w').write(str(int(IDLE_MIN)))


def _note(msg):
    try:
        open(LOGF, 'a').write(f'{time.ctime()} {msg}\n')
    except Exception:
        pass


def _is_idle():
    try:
        if float(open('/proc/loadavg').read().split()[1]) >= IDLE_LOAD:
            return False
    except Exception:
        return False  # 알 수 없으면 쉬지 않는 것으로 본다
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        if any(int(x) >= 5 for x in out.split()):
            return False
    except Exception:
        pass  # GPU가 없는 runtime: load만 본다
    return True


def _watch():
    idle = 0
    while True:
        try:
            dl = float(open(DL).read().strip())
        except Exception:
            dl = None
        try:
            idle_min = int(open(IDLE).read().strip())
        except Exception:
            idle_min = 0
        idle = idle + 1 if _is_idle() else 0
        why = None
        if dl is not None and time.time() > dl:
            why = 'deadline'
        elif idle_min > 0 and idle >= idle_min:
            why = f'idle {idle} min'
        if why:
            _note(f'unassign ({why})')
            runtime.unassign()
            return
        time.sleep(60)


if not any(t.name == 'selfstop' for t in threading.enumerate()):
    threading.Thread(target=_watch, name='selfstop', daemon=True).start()
print('SELFSTOP armed', SECONDS, 's at', time.ctime(time.time() + SECONDS), 'idle_min', IDLE_MIN,
      'threads', sum(t.name == 'selfstop' for t in threading.enumerate()))
