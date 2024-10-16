import torch
import torch.optim as Opt

class Adam_Opt(Opt.Adam):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(Adam_Opt, self).__init__(model.parameters(), lr=lr_start)

class SGD_Opt(Opt.SGD):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(SGD_Opt, self).__init__(model.parameters(), lr=lr_start)

class Adagrad_Opt(Opt.Adagrad):
    def __init__(self, model:torch.nn.Module, args):
        lr_start = args.lr_start
        super(Adagrad_Opt, self).__init__(model.parameters(), lr=lr_start)

def get_optimizer(args, model):
    opt_fns = {
        'adam': Adam_Opt,
        'sgd': SGD_Opt,
        'adagrad': Adagrad_Opt,
    }
    opt_fn = opt_fns.get(args.optimizer, "Invalid Optimizer")
    opt = opt_fn(model.module, args)
    return opt