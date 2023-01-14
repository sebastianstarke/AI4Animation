# AdamW optimizer and cosine learning rate annealing with restarts

This repository contains an implementation of AdamW optimization algorithm and cosine learning rate scheduler described in ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101). AdamW implementation is straightforward and does not differ much from existing Adam implementation for PyTorch, except that it separates weight decaying from batch gradient calculations.
Cosine annealing scheduler with restarts allows model to converge to a (possibly) different local minimum on every restart and normalizes weight decay hyperparameter value according to the length of restart period.
Unlike schedulers presented in standard PyTorch scheduler suite this scheduler adjusts optimizer's learning rate not on every epoch, but on every batch update, according to the paper.
## Cyclical Learning Rates
Besides ["cosine"](https://www.google.com/search?q=(cos(x%2Fpi)%2B1)%2F2) and ["arccosine"](https://www.google.com/search?q=arccos(2*x-1)%2Fpi) policies (`arccosine` has steeper profile at the limiting points), there are ["triangular"](https://www.google.com/search?q=1-abs(x*2-1)), `triangular2` and `exp_range`, which implement policies proposed in ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186).
The ratio of increasing and decreasing phases for triangular policy could be adjusted with `triangular_step` parameter. Minimum allowed lr is adjusted by `min_lr` parameter.

* `triangular` schedule is enabled by passing `policy="triangular"` parameter.
* `triangular2` schedule reduces maximum lr by half on each restart cycle and is enabled by passing `policy="triangular2"` parameter, or by combining parameters `policy="triangular", eta_on_restart_cb=ReduceMaxLROnRestart(ratio=0.5)`. The `ratio` parameter regulates the factor by which lr is scaled on each restart.
* `exp_range` schedule is enabled by passing `policy="exp_range"` parameter. It exponentially scales maximum lr depending on iteration count. The base of exponentiation is set by `gamma` parameter.

These schedules could be combined with shrinking/expanding restart periods, weight decay normalization and could be used with AdamW and other PyTorch optimizers.

# Example:
```python
    batch_size = 32
    epoch_size = 1024
    model = resnet()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=5, t_mult=1.2, policy="cosine")
    for epoch in range(100):
        scheduler.step()
        train_for_every_batch(...)
            ...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()
        validate(...)
```        
