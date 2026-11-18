"""E45 smoke test: planning rollout이 만드는 forecaster feature가 학습 column과 같은 정의인지 확인한다.

같은 episode의 실제 frame 3장(t-10, t-5, t)을 planning처럼 history로 넣고 `CFJEPA.rollout`이 만든 info['forecast']를,
`train_int_frozen.forecast_columns`가 같은 frame에서 만든 depth별 값과 비교한다. 정의가 어긋나면(index를 한 칸 밀었거나
depth를 잘못 셌다면) 큰 차이가 난다. column은 float16이므로 작은 차이는 허용한다. gate를 끄면 cost가 달라지는지도 본다.

사용: python scripts/colab/check_forecast.py --ckpt e45_fc/weights_best.pt --data r1dbg.h5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--episodes', type=int, default=5)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm

    from analyze_gate import to_model_pixels
    from iawm.cfjepa import history_get_cost
    from train_int_frozen import FC_TRUNC, FS, forecast_columns, load_rows

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    home = Path(swm.data.utils.get_cache_dir())
    m = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    pr = m.predictor
    assert pr.forecaster is not None and pr.int_oracle_key == 'forecast', 'forecast 입력 모델이 아니다'
    rows, off, ep_len = load_rows(m, home / 'datasets' / args.data, device, None, 'forecast')
    cols = forecast_columns(pr.forecaster, rows)
    H, S, n_steps = pr.num_frames, 4, 4
    T = H + n_steps
    A = m.action_encoder.input_dim
    worst = 0.0
    with h5py.File(home / 'datasets' / args.data, 'r') as f:
        eps = [e for e in range(len(ep_len)) if ep_len[e] > FS * (H + n_steps + 2)][:args.episodes]
        for e in eps:
            r0 = int(off[e]) + FS * (H - 1) + 3
            hist = [r0 - FS * (H - 1 - j) for j in range(H)]
            px = to_model_pixels(f['pixels'][hist][None], device)
            goal = to_model_pixels(f['pixels'][[r0 + FS * n_steps]][None], device)
            info = {'pixels': px[:, None].expand(1, S, *px.shape[1:]).clone(),
                    'goal': goal[:, None].expand(1, S, *goal.shape[1:]).clone(),
                    'hist_action': torch.zeros(1, S, H - 1, A, device=device),
                    'action': torch.zeros(1, S, H, A, device=device)}  # policy가 넣는 key (get_cost가 goal에서 뺀다)
            cand = torch.randn(1, S, T - (H - 1), A, device=device)
            with torch.no_grad():
                cost_on = history_get_cost(m, info, cand)
            F = info['forecast'][0, 0]
            assert F.shape[0] == T, (F.shape, T)
            for i in range(T):
                if i < H - 1:  # history 첫 두 frame: 앞 frame이 잘린 column
                    c, row = FC_TRUNC[i], hist[i]
                else:          # depth i - H + 2: 실제 history 끝(r0)에서 굴린 값
                    d = i - H + 2
                    c, row = d - 1, r0 + FS * (d - 1)
                worst = max(worst, (F[i].float() - cols[c, row].float()).abs().max().item())
            mode = pr.gate_mode
            pr.gate_mode = 'off'
            try:
                info2 = {k: info[k] for k in ('pixels', 'goal', 'hist_action', 'action')}
                with torch.no_grad():
                    cost_off = history_get_cost(m, info2, cand)
            finally:
                pr.gate_mode = mode
            print(f'episode {e}: cost on {cost_on[0].tolist()} off {cost_off[0].tolist()}', flush=True)
    scale = cols.float().abs().max().item()
    ok = worst < 1e-2 * max(1.0, scale)  # float16 column의 반올림 오차보다 크면 정의가 어긋난 것이다
    print(f"FORECAST CHECK {'OK' if ok else 'BAD'}: max |plan - column| {worst:.2e} (column max abs {scale:.3f}), "
          f"features {tuple(info['forecast'].shape)}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
