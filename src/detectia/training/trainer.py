"""Trainer: encapsula o loop de treino completo.

Responsabilidades:
  - Loop principal (épocas)
  - Treino e validação por época
  - AMP (mixed precision)
  - Checkpoint do melhor modelo
  - Early stopping
  - Logging em TensorBoard
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from tqdm import tqdm

from .metrics import AverageMeter


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "runs/exp",
        use_amp: bool = True,
        early_stopping_patience: int = 3,
        monitor: str = "val_auc",
        monitor_mode: str = "max",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss pra binário (combina sigmoid + BCE de forma estável)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # AMP (mixed precision)
        self.use_amp = use_amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
        # Métricas (torchmetrics gerencia estado interno)
        self.train_acc = BinaryAccuracy().to(device)
        self.val_acc = BinaryAccuracy().to(device)
        self.val_auc = BinaryAUROC().to(device)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / "tb")
        
        # Early stopping
        self.patience = early_stopping_patience
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.best_metric = float("-inf") if monitor_mode == "max" else float("inf")
        self.epochs_no_improve = 0
    
    def fit(self, num_epochs: int):
        """Loop principal: treina por num_epochs (ou até early stopping)."""
        print(f"\n{'='*60}")
        print(f"Treino iniciado: {num_epochs} épocas | device={self.device} | AMP={self.use_amp}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_auc = self._validate(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            elapsed = time.time() - t0
            print(
                f"Época {epoch:>2}/{num_epochs} | "
                f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | "
                f"{elapsed:.1f}s"
            )
            
            # TensorBoard logging
            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)
            self.writer.add_scalar("acc/train", train_acc, epoch)
            self.writer.add_scalar("acc/val", val_acc, epoch)
            self.writer.add_scalar("auc/val", val_auc, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            
            # Checkpoint + early stopping
            current_metric = {
                "val_auc": val_auc, "val_acc": val_acc, "val_loss": val_loss,
            }[self.monitor]
            improved = (
                current_metric > self.best_metric
                if self.monitor_mode == "max"
                else current_metric < self.best_metric
            )
            
            if improved:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, current_metric)
                print(f"  ✓ Melhor {self.monitor}: {current_metric:.4f} — checkpoint salvo")
            else:
                self.epochs_no_improve += 1
                print(f"  ✗ {self.monitor} não melhorou ({self.epochs_no_improve}/{self.patience})")
            
            if self.epochs_no_improve >= self.patience:
                print(f"\n=== Early stopping na época {epoch} ===")
                break
        
        self.writer.close()
        print(f"\n=== Encerrado. Melhor {self.monitor}: {self.best_metric:.4f} ===")
    
    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Uma época de treino. Retorna (loss média, acc da época)."""
        self.model.train()
        loss_meter = AverageMeter()
        self.train_acc.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Train ep{epoch}", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            # Labels: int -> float, shape [B] -> [B, 1] (mesma shape do output do modelo)
            labels = labels.to(self.device, non_blocking=True).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward com AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            # Backward com AMP (scaler escala loss pra evitar underflow em FP16)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Métricas
            loss_meter.update(loss.item(), images.size(0))
            preds = (torch.sigmoid(logits.detach()).squeeze() > 0.5).long()
            self.train_acc.update(preds, labels.squeeze().long())
            
            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{self.train_acc.compute().item():.4f}",
            )
        
        return loss_meter.avg, self.train_acc.compute().item()
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[float, float, float]:
        """Validação. Retorna (loss média, acc, auc)."""
        self.model.eval()
        loss_meter = AverageMeter()
        self.val_acc.reset()
        self.val_auc.reset()
        
        for images, labels in tqdm(self.val_loader, desc=f"Val ep{epoch}", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float().unsqueeze(1)
            
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            loss_meter.update(loss.item(), images.size(0))
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).long()
            
            self.val_acc.update(preds, labels.squeeze().long())
            self.val_auc.update(probs, labels.squeeze().long())
        
        return loss_meter.avg, self.val_acc.compute().item(), self.val_auc.compute().item()
    
    def _save_checkpoint(self, epoch: int, metric: float):
        """Salva o estado do modelo + optimizer pra continuação."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metric": metric,
            "monitor": self.monitor,
        }
        torch.save(ckpt, self.output_dir / "best.pt")