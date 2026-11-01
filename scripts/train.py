"""World model 학습 script. LeWM의 train.py(github.com/lucas-maes/le-wm, MIT)를 고쳐 쓴다.

- `model=cfjepa`(기본): CF-JEPA loss로 학습한다.
- `model=lewm`: LeWM loss(forward + SIGReg)로 학습한다. distractor 조건 baseline용이다.
- `distractor.num=N`: 학습 이미지에 사각형 N개를 덧그린다(0이면 끈다). 논문은 N=2다.

사용 예:
    python scripts/train.py                         # CF-JEPA, TwoRoom, N=2
    python scripts/train.py model=lewm output_model_name=lewm_n2
    python scripts/train.py trainer.max_epochs=1 +trainer.limit_train_batches=50   # smoke test
"""

import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.wm.loss import SIGReg

from iawm.distractors import DistractorConfig, overlay, sample_params

sys.path.insert(0, str(ROOT / 'scripts'))
from train_utils import SaveCkptCallback, get_column_normalizer, get_img_preprocessor  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3)


def add_distractors(batch, cfg):
    """batch['pixels'] (B, T, C, H, W, ImageNet 정규화)에 사각형을 덧그린다."""
    n = cfg.distractor.num
    if n <= 0:
        return
    pixels = batch['pixels']
    dcfg = DistractorConfig(num=n, img_size=pixels.size(-1))
    params = sample_params(pixels.size(0), dcfg, device=pixels.device)
    # 연속한 frame 사이에 frameskip만큼 env step이 흐른다.
    t = torch.arange(pixels.size(1), device=pixels.device).float() * cfg.data.dataset.frameskip
    batch['pixels'] = overlay(pixels, params, t, dcfg, IMAGENET_MEAN, IMAGENET_STD)


def masked_mse(pred, target, mask):
    err = (pred.float() - target.float()).pow(2).mean(-1)  # (B, T)
    return (err * mask).sum() / mask.sum().clamp(min=1)


def world_model_forward(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    # 에피소드 경계에서 action이 NaN이다. inverse/adversarial loss에서는 그 step을 뺀다.
    action_valid = (~torch.isnan(batch['action']).any(-1)).float()  # (B, T)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    add_distractors(batch, cfg)
    output = self.model.encode(batch)
    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    tgt_emb = emb[:, n_preds:]

    output['fwd_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    loss = output['fwd_loss'] + cfg.loss.sigreg_weight * output['sigreg_loss']

    if cfg.model_kind == 'cfjepa':
        z_c, z_u = self.model.split(emb)
        act = batch['action'][:, :-1]  # action[t]: frame t -> t+1
        mask = action_valid[:, :-1]
        output['inv_loss'] = masked_mse(self.model.inverse(z_c[:, :-1], z_c[:, 1:]), act, mask)
        # φ는 이 loss를 줄이도록, encoder는 gradient reversal 때문에 키우도록 학습된다.
        # 값이 action 분산(정규화했으므로 약 1)에 가까우면 z^u에 action 정보가 없다는 뜻이다.
        output['adv_loss'] = masked_mse(self.model.adversary(z_u[:, :-1], z_u[:, 1:]), act, mask)
        loss = loss + cfg.loss.inv_weight * output['inv_loss'] + cfg.loss.adv_weight * output['adv_loss']

    # ---- [interaction gate] 강연우 추가 (11/1) ----------------------------------------
    # predictor가 InteractionGatedPredictor(src/iawm/cfjepa_gate.py)이면 sparsity ρ·E[α]를
    # loss에 더하고 α 통계를 로깅한다. ρ는 model config의 gate_sparsity_weight다.
    # last_alpha가 없는 모델(CF-JEPA, LeWM)은 이 블록을 건너뛰므로 기존 학습과 같다.
    alpha = getattr(self.model.predictor, 'last_alpha', None)
    if alpha is not None:
        alpha = alpha.float()
        output['gate_sparsity_loss'] = alpha.mean()
        loss = loss + cfg.model.get('gate_sparsity_weight', 0.0) * output['gate_sparsity_loss']
        self.log_dict(
            {
                f'{stage}/gate_alpha_mean': alpha.mean().detach(),
                f'{stage}/gate_alpha_max': alpha.max().detach(),
                f'{stage}/gate_open_ratio': (alpha > 0.5).float().mean().detach(),
            },
            on_step=True, sync_dist=True,
        )
    # ---- [interaction gate] 추가 끝 ---------------------------------------------------

    output['loss'] = loss
    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True, sync_dist=True,
    )
    return output


@hydra.main(version_base=None, config_path='../configs/train', config_name='cfjepa')
def run(cfg):
    with open_dict(cfg):
        cfg.model_kind = 'cfjepa' if cfg.model._target_.endswith('CFJEPA') else 'lewm'
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=os.environ.get('LOCAL_DATASET_DIR'), **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim('action')
        cfg.model.action_encoder.input_dim = action_dim
        if cfg.model_kind == 'cfjepa':
            cfg.model.action_dim = action_dim
    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    pl.seed_everything(cfg.seed)
    world_model = hydra.utils.instantiate(cfg.model)
    n_params = sum(p.numel() for p in world_model.parameters())
    print(f'model={cfg.model_kind} params={n_params / 1e6:.2f}M distractors={cfg.distractor.num}')

    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg_kwargs),
        forward=partial(world_model_forward, cfg=cfg),
        optim={
            'model_opt': {
                'modules': 'model',
                'optimizer': dict(cfg.optimizer),
                'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
                'interval': 'epoch',
            },
        },
    )

    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.output_model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'train_config.yaml')

    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        logger = CSVLogger(save_dir=str(run_dir), name='logs')

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1)],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )()


if __name__ == '__main__':
    run()
