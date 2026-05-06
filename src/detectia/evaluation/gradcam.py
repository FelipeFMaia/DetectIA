"""Grad-CAM para visualização de regiões importantes na decisão do modelo.

Implementação manual (sem dependência externa) pra fins didáticos.
Existem libs prontas (pytorch-grad-cam) se quiser usar em projetos sérios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM via hooks na camada alvo.
    
    Pra ResNet18, a camada natural é `model.layer4` — último bloco residual
    antes do Global Average Pooling. Mantém resolução espacial 7×7
    (com input 224×224) que é boa pra localizar regiões grosseiras.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Hooks: capturamos ativações no forward e gradientes no backward
        self.hook_fwd = target_layer.register_forward_hook(self._save_activations)
        self.hook_bwd = target_layer.register_full_backward_hook(self._save_gradients)
    
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: str = "predicted",
    ) -> tuple:
        """
        Computa heatmap pra um input.
        
        Args:
            input_tensor: [1, 3, H, W] — imagem normalizada (com .to(device))
            target_class: "predicted" (default), "real" ou "ai"
                "predicted" → backward na classe que o modelo prevê
                "real"      → backward em logit (sempre olha pra Real)
                "ai"        → backward em -logit (sempre olha pra IA)
        
        Returns:
            (heatmap [H, W] em [0,1], prob_real float)
        """
        self.model.eval()
        self.model.zero_grad()
        
        logit = self.model(input_tensor)             # [1, 1]
        prob = torch.sigmoid(logit).item()
        
        # Decide qual logit usar pro backward
        if target_class == "real":
            target = logit.squeeze()
        elif target_class == "ai":
            target = -logit.squeeze()
        else:  # "predicted"
            target = logit.squeeze() if prob > 0.5 else -logit.squeeze()
        
        target.backward()
        
        # Pesos = GAP dos gradientes (importância média de cada canal)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # CAM = soma ponderada das ativações
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)  # só contribuições positivas
        
        # Upsample pra tamanho da imagem original
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        
        # Normaliza pra [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, prob
    
    def remove_hooks(self):
        """Limpa os hooks pra evitar vazamento de memória."""
        self.hook_fwd.remove()
        self.hook_bwd.remove()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()