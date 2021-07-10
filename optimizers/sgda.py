import torch
from torch.optim.optimizer import Optimizer


class SGDA(Optimizer):
    r"""Implements SGDA algorithm.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=0.001, eps=1e-8, weight_decay=0):
        assert lr > 0, "Invalid learning rate: {}".format(lr)
        assert eps > 0, "Invalid epsilon value: {}".format(eps)
        assert weight_decay >= 0.0, "Invalid weight_decay value: {}".format(weight_decay)
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                state = self.state[p]

                p_ = torch.norm(p, p=2)
                grad_ = torch.norm(grad, p=2)
                delta = (grad_/p_)
                lr_ = group['lr']/delta
                
                p.add_(grad, alpha=-lr_)

        return loss