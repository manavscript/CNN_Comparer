from training import Trainer
from train_loader import NNDataLoader
from model.nn import SimpleNeuralNetwork, SimpleCNN, MnistResNet
import torch.optim 
from torch.nn import CrossEntropyLoss
import argparse
import numpy as np
import time
from torchvision import models

def infer_time(test, model, device, single = False):
    start_time = time.time()
    for X, y in test:
        X, y = X.to(device), y.to(device)
        model.to(device)
        pred = model(X)
        if single: break
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser Args')
    parser.add_argument('-n','--name', help='Name of Model', default=None)
    parser.add_argument('-p','--pretrained', help='pretrained', type=bool, default=False)
    parser.add_argument('-m','--model', help='Model File Path', default=None)
    parser.add_argument('-e','--epochs', help='Epochs', default=10)
    parser.add_argument('-d','--dataset', help='Dataset', default=10)

    args = parser.parse_args()

    lr = 1e-3
    epochs = int(args.epochs)
    reg = 1e-5
    batch_size = 32

    data_load_obj = NNDataLoader()
    if args.dataset == "FashionMnist":
        data_load_obj.fasion_mnist()
    else:
        data_load_obj.tiny_imagenet()

    train, val, test = data_load_obj.create_dataloader()

    if args.dataset == "FashionMnist":
        if args.name == "Simple":
            model = SimpleNeuralNetwork()
        elif args.name == "S_CNN":
            model = SimpleCNN()
        elif args.name == "Resnet":
            model = MnistResNet()
        else:
            model = SimpleNeuralNetwork()
    else:
        model = models.resnet50(pretrained=True)

    print(args)
    if args.model is not None and not args.pretrained:
        model.load_state_dict(torch.load(args.model))

    trainer = Trainer(lr, epochs, reg, batch_size)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    loss_fn = CrossEntropyLoss()
    
    # if not args.pretrained:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if (epochs != 0): trainer.train(train, val, model, loss_fn, optimizer, device, name=str(args.name))
    
    print(len(train), len(train.dataset))
    print(len(val), len(val.dataset))
    print(len(test), len(test.dataset))
    
    trainer.evaluate(model, test, device, loss_fn)

    trials = []
    trials_sb = []
    for i in range(7):
        trials.append(infer_time(test, model, device))
        trials_sb.append(infer_time(test, model, device, single = True))
    
    
    print(f"Simple {args.name} 1 Batch Time: " + str(np.mean(trials_sb)) + " " + str(np.var(trials_sb)))
    print(f"Simple {args.name} ~300 Batch Time: " + str(np.mean(trials)) + " " + str(np.var(trials)))

    #Simple DNN: 88.4% -> 9
    #Simple CNN: 90.7-> 5

    # Simple None 1 Batch Time: 0.0022211415427071707 5.6840567409306785e-08
    # Simple None ~300 Batch Time: 0.6003268446241107 0.001857126686944384

    # Simple Resnet 1 Batch Time: 0.03076314926147461 4.3811948752850837e-05
    # Simple Resnet ~300 Batch Time: 12.785346610205513 5.231835632508606

    #Simple S_CNN 1 Batch Time: 0.0023492404392787387 1.5794542526478202e-08
    #Simple S_CNN ~300 Batch Time: 0.6860007217952183 1.4808585544823455e-05
