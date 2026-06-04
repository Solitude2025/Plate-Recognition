import torch

def create_optimizer(model, lr=1e-3, weight_decay=5e-4):
    params = [
        {"params": [p for n, p in model.named_parameters() if 'bias' not in n], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if 'bias' in n], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(params, lr=lr)
    return optimizer

def create_scheduler(optimizer, total_epochs, warmup_epochs=3, min_lr=1e-6):

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max(min_lr / optimizer.defaults['lr'], 0.5 * (1.0 + torch.cos(torch.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


