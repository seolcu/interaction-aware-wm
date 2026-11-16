# Colab 쪽 이중 안전장치 (colab-guard arm이 부른다). kernel 안 thread가 1분마다 /content/selfstop_deadline
# (epoch 초)을 읽고, 그 시각이 지나면 runtime을 스스로 반납한다. 로컬 컴퓨터가 꺼져 있어도 세션이 꺼진다.
# 다시 부르면 기한 파일만 바꾼다(연장). thread는 하나만 띄운다.
SECONDS = 4 * 3600
import threading, time
from google.colab import runtime

DL = '/content/selfstop_deadline'
open(DL, 'w').write(str(int(time.time() + SECONDS)))

def _watch():
    while True:
        try:
            dl = float(open(DL).read().strip())
        except Exception:
            dl = None
        if dl is not None and time.time() > dl:
            try:
                open('/content/selfstop.log', 'a').write(f'{time.ctime()} unassign\n')
            except Exception:
                pass
            runtime.unassign()
            return
        time.sleep(60)

if not any(t.name == 'selfstop' for t in threading.enumerate()):
    threading.Thread(target=_watch, name='selfstop', daemon=True).start()
print('SELFSTOP armed', SECONDS, 's at', time.ctime(time.time() + SECONDS),
      'threads', sum(t.name == 'selfstop' for t in threading.enumerate()))
