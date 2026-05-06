"""Métricas auxiliares de treino."""


class AverageMeter:
    """Mantém média móvel de uma métrica (ex: loss média do batch)."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0.0
        self.count = 0
    
    def update(self, value: float, n: int = 1):
        """value: valor do batch. n: quantidade de amostras no batch."""
        self.sum += value * n
        self.count += n
    
    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0