"""HF LeWM checkpoint의 ViT key를 transformers v5 이름으로 바꿔 저장한다.

quentinll/lewm-* 의 weights.pt는 transformers v4의 ViTModel key
(encoder.encoder.layer.N.attention.attention.query ...)로 저장되어 있다.
stable-worldmodel이 쓰는 transformers v5는 같은 layer를 다른 이름
(encoder.layers.N.attention.q_proj ...)으로 부르므로 그대로는 load_state_dict가 실패한다.
값은 바꾸지 않고 key 이름만 옮긴다.

사용: python fix_lewm_keys.py <hf_repo> <out_name>
  예: python fix_lewm_keys.py quentinll/lewm-tworooms lewm-tworooms
  결과: $STABLEWM_HOME/checkpoints/<out_name>/{weights.pt, config.json}
        -> swm.wm.utils.load_pretrained('<out_name>') 로 읽는다.
"""

import re
import shutil
import sys
from pathlib import Path

import stable_worldmodel as swm
import torch
from huggingface_hub import hf_hub_download

RENAMES = [
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_", r"encoder.layers.\1.layernorm_"),
]

repo, out_name = sys.argv[1], sys.argv[2]
out = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), out_name)
out.mkdir(parents=True, exist_ok=True)

sd = torch.load(hf_hub_download(repo, "weights.pt"), map_location="cpu")
new = {}
for k, v in sd.items():
    for pat, rep in RENAMES:
        k2 = re.sub(pat, rep, k)
        if k2 != k:
            k = k2
            break
    new[k] = v
torch.save(new, out / "weights.pt")
shutil.copy(hf_hub_download(repo, "config.json"), out / "config.json")

model = swm.wm.utils.load_pretrained(out_name)  # strict load로 검증
n = sum(p.numel() for p in model.parameters()) / 1e6
print(f"OK {out} ({type(model).__name__}, {n:.2f}M params)")
