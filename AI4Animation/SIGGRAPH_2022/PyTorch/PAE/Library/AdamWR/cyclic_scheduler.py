from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch


class ReduceMaxLROnRestart:
    def __init__(self, ratio=0.75):
        self.ratio = ratio

    def __call__(self, eta_min, eta_max):
        return eta_min, eta_max * self.ratio


class ExpReduceMaxLROnIteration:
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, eta_min, eta_max, iterations):
        return eta_min, eta_max * self.gamma ** iterations


class CosinePolicy:
    def __call__(self, t_cur, restart_period):
        return 0.5 * (1. + math.cos(math.pi *
                                    (t_cur / restart_period)))


class ArccosinePolicy:
    def __call__(self, t_cur, restart_period):
        return (math.acos(max(-1, min(1, 2 * t_cur
                                      / restart_period - 1))) / math.pi)


class TriangularPolicy:
    def __init__(self, triangular_step=0.5):
        self.triangular_step = triangular_step

    def __call__(self, t_cur, restart_period):
        inflection_point = self.triangular_step * restart_period
        point_of_triangle = (t_cur / inflection_point
                             if t_cur < inflection_point
                             else 1.0 - (t_cur - inflection_point)
                             / (restart_period - inflection_point))
        return point_of_triangle


class CyclicLRWithRestarts(_LRScheduler):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will expand/shrink
        policy: ["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
        min_lr: minimum allowed learning rate
        verbose: print a message on every restart
        gamma: exponent used in "exp_range" policy
        eta_on_restart_cb: callback executed on every restart, adjusts max or min lr
        eta_on_iteration_cb: callback executed on every iteration, adjusts max or min lr
        triangular_step: adjusts ratio of increasing/decreasing phases for triangular policy


    Example:
        >>> scheduler = CyclicLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, verbose=False,
                 policy="cosine", policy_fn=None, min_lr=1e-7,
                 eta_on_restart_cb=None, eta_on_iteration_cb=None,
                 gamma=1.0, triangular_step=0.5):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('minimum_lr', min_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))

        self.base_lrs = [group['initial_lr'] for group
                         in optimizer.param_groups]

        self.min_lrs = [group['minimum_lr'] for group
                        in optimizer.param_groups]

        self.base_weight_decays = [group['weight_decay'] for group
                                   in optimizer.param_groups]

        self.policy = policy
        self.eta_on_restart_cb = eta_on_restart_cb
        self.eta_on_iteration_cb = eta_on_iteration_cb
        if policy_fn is not None:
            self.policy_fn = policy_fn
        elif self.policy == "cosine":
            self.policy_fn = CosinePolicy()
        elif self.policy == "arccosine":
            self.policy_fn = ArccosinePolicy()
        elif self.policy == "triangular":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
        elif self.policy == "triangular2":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)
        elif self.policy == "exp_range":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.iteration = 0
        self.total_iterations = 0

        self.t_mult = t_mult
        self.verbose = verbose
        self.restart_period = math.ceil(restart_period)
        self.restarts = 0
        self.t_epoch = -1
        self.epoch = -1

        self.eta_min = 0
        self.eta_max = 1

        self.end_of_period = False
        self.batch_increments = []
        self._set_batch_increment()

    def _on_restart(self):
        if self.eta_on_restart_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min,
                                                                self.eta_max)

    def _on_iteration(self):
        if self.eta_on_iteration_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min,
                                                                  self.eta_max,
                                                                  self.total_iterations)

    def get_lr(self, t_cur):
        eta_t = (self.eta_min + (self.eta_max - self.eta_min)
                 * self.policy_fn(t_cur, self.restart_period))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))

        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr
               in zip(self.base_lrs, self.min_lrs)]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
            self.end_of_period = True

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart {} at epoch {}".format(self.restarts + 1,
                                                      self.last_epoch))
            self.restart_period = math.ceil(self.restart_period * self.t_mult)
            self.restarts += 1
            self.t_epoch = 0
            self._on_restart()
            self.end_of_period = False

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()

    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.batch_step()

    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            self.iteration += 1
            self.total_iterations += 1
        except (IndexError):
            raise StopIteration("Epoch size and batch size used in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
