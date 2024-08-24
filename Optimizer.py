
class Lscheduler:
    def __init__(self,optimizer,warmups,model_dimension):
        self.model_dim = model_dimension
        self.opt = optimizer
        self.warmup=warmups
        self.current_step=0

    def step(self):
        """Takes a step in the scheduler and updates the learning rate
            for all parameter groups."""
        self.current_step+=1
        lr = self.get_lr()
        for param in self.opt.param_groups:
            param['lr'] = lr
        
        self.opt.step()

    def get_lr(self):
        """Gets the current learning rate based on the current step"""
        learning_rate = (self.model_dim**(-0.5))*min(self.current_step**(-0.5),(self.current_step*(self.warmup**(-0.5))))
        return learning_rate

    def zero_grad(self):
        """Zeros the gradients for every param group."""
        self.opt.zero_grad()