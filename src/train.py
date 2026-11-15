"""Training loop with train and validation cross-entropy per epoch."""
import torch
import torch.nn as nn


def make_optimizer(model, cfg):
    name = cfg["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"]
        )
    raise ValueError(f"Unknown optimizer {cfg['optimizer']!r} (use 'adam' or 'sgd')")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One pass over the training data. Returns mean cross-entropy per example."""
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        # loss is a batch mean, so multiply by batch size to get a correct overall mean.
        total_loss += loss.item() * len(xb)
        total_n += len(xb)
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Mean cross-entropy without updating weights."""
    model.eval()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        total_loss += loss.item() * len(xb)
        total_n += len(xb)
    return total_loss / total_n


def fit(model, train_loader, val_loader, cfg, device):
    """Train for cfg['epochs'] epochs. Returns a list of {epoch, train_loss, val_loss}."""
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader is not None else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:3d} | train CE = {train_loss:.4f} | val CE = {val_loss:.4f}")
    return history
