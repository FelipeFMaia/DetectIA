"""CLI de treino: carrega config YAML, monta tudo, chama Trainer.fit()."""

import argparse
from pathlib import Path

import torch
import yaml

from detectia.data.loaders import make_loaders
from detectia.models.classifier import build_model, count_parameters
from detectia.training.trainer import Trainer


def build_optimizer(model_params, opt_cfg):
    name = opt_cfg["name"].lower()
    if name == "adam":
        return torch.optim.Adam(
            model_params,
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0),
        )
    raise ValueError(f"Optimizer desconhecido: {name}")


def build_scheduler(optimizer, sched_cfg, num_epochs):
    if sched_cfg is None:
        return None
    name = sched_cfg["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=sched_cfg.get("eta_min", 0),
        )
    raise ValueError(f"Scheduler desconhecido: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 60)
    print(f"Config carregada: {args.config}")
    print("=" * 60)
    print(yaml.dump(cfg, default_flow_style=False))
    
    # Reprodutibilidade
    torch.manual_seed(cfg["experiment"]["seed"])
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DataLoaders
    print("=== Criando DataLoaders ===")
    train_loader, val_loader = make_loaders(
        source=cfg["data"]["source"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_samples_train=cfg["data"].get("max_samples_train"),
        max_samples_val=cfg["data"].get("max_samples_val"),
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")
    
    # Modelo
    print("=== Construindo modelo ===")
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    )
    p = count_parameters(model)
    print(f"Params: total={p['total']:,} | treináveis={p['trainable']:,}\n")
    
    # Optimizer (só nos params treináveis — importante quando freeze_backbone=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable_params, cfg["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer, cfg["training"].get("scheduler"), cfg["training"]["epochs"]
    )
    
    # Output dir
    output_dir = Path(cfg["logging"]["output_dir"]) / cfg["experiment"]["name"]
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=str(output_dir),
        use_amp=cfg["training"]["amp"],
        early_stopping_patience=cfg["training"]["early_stopping"]["patience"],
        monitor=cfg["training"]["early_stopping"]["monitor"],
        monitor_mode=cfg["training"]["early_stopping"]["mode"],
    )
    
    trainer.fit(num_epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()