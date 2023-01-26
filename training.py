import torch

class Trainer():
    def __init__(self, lr, epochs, reg, batch_size) -> None:
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.batch_size = batch_size

    def train(self, dataloader, val_dataloader, model, loss_fn, optimizer, device, name):
        epoch_loss = {}
        epoch_accuracy = {}
        for ep in range(self.epochs):
            batch_loss = {}
            batch_accuracy = {}
            loss = 0.0
            correct = 0
            size = 0
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                model.to(device)
                
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
                x_size = len(X)
                
                correct += _correct
                
                # Updating loss_batch and batch_accuracy
                batch_loss[batch] = loss.item()
                batch_accuracy[batch] = _correct/x_size
                loss += loss.item()
                
                size += x_size
                
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}]")
            
            torch.save(model.state_dict(), f"history/model_{ep}_{name}.pth")
            print("Saved PyTorch Model State")

            correct/=size
            print(f"Train Accuracy: {(100*correct):>0.1f}%")
            epoch_accuracy[ep] = correct
            epoch_loss[ep] = loss
            self.evaluate(model, val_dataloader, device, loss_fn)
            
        return epoch_loss, epoch_accuracy

    def evaluate(self, model, dataloader, device, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            print(next(iter(dataloader)))
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                model.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")