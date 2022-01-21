import numpy as np
import torch
import os
import json
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, monitor="loss", patience=7, verbose=False, delta=0.0001, path='checkpoint.pt',save_model=True, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor_metric_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_model = save_model
    def __call__(self, monitor_metric, model, **kwarg):
        if self.monitor == "loss":
            score = -monitor_metric
        else:
            score = monitor_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(monitor_metric, model, **kwarg)
        elif score < self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}, {self.best_score}')
            #if self.counter >= self.patience:
            #    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(monitor_metric, model, **kwarg)
            self.counter = 0

    def save_checkpoint(self, monitor_metric, model, **kwarg):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation {self.monitor} is improved ({self.monitor_metric_min:.6f} --> {monitor_metric:.6f}).  Saving model ...')
        if self.save_model:
            torch.save(model.state_dict(), os.path.join(self.path, "saved_model.pt"))
        with open(os.path.join(self.path, "performance.json"), "w") as f:
            json.dump(kwarg["performance"], f )
          
        self.monitor_metric_min = monitor_metric

def save_model_layer_bame(model, saved_path):
    layers=[]
    with open(saved_path, "w") as f:
        for (name, _) in model.named_modules():
            layers.append(name)
            f.write(name+"\n")
        
    json.dump( layers, open(saved_path+".json", "w") )


    