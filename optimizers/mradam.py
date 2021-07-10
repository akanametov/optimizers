import math
import torch
from torch.optim.optimizer import Optimizer


class MRAdam(Optimizer):
    r"""Implements MRAdam algorithm.

    It has been proposed in `MRAdam`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        gammas (Tuple[int, int], optional): coefficients used for choosing
            bounds for optimizers (default: (2, 8))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gammas=(2, 8), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 1 <= gammas[0] < 16:
            raise ValueError("Invalid gamma parameter at index 0: {}".format(gammas[0]))
        if not 2 <= gammas[1] < 20:
            raise ValueError("Invalid gamma parameter at index 1: {}".format(gammas[1]))
        if not gammas[0] < gammas[1]:
            raise ValueError("Invalid gamma parameters: {}".format(gammas))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, gammas=gammas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MRdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    beta1, beta2 = group['betas']
                    state['step'] = 0
                    state['sma_max'] = (2 /(1 - beta2)) - 1
                    state['sma'] = 0
                    state['ro'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state['m'], state['v']
                if amsgrad:
                    max_v = state['max_v']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                state['sma'] = state['sma_max'] - 2*state['step']*(beta2**state['step'])/bias_correction2
                state['ro'] = math.sqrt(abs(((state['sma'] - 4)*(state['sma'] - 2)*state['sma_max'])\
                                            /((state['sma_max'] - 4)*(state['sma_max'] - 2)*state['sma'])))

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_v, v, out=max_v)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    
                gamma1, gamma2 = group['gammas']
                
                if state['sma'] <= 2**gamma1:
                    step_size = group['lr'] / bias_correction1
                    p.add_(m, alpha=-step_size)
                    
                elif 2**gamma1 < state['sma'] <= 2**gamma2:
                    step_size = group['lr']
                    p.addcdiv_(grad, denom, value=-step_size)
                    
                else:
                    step_size = group['lr']*state['ro'] / bias_correction1
                    p.addcdiv_(m, denom, value=-step_size)

        return loss