class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        """
        早停机制
        Args:
            patience (int): 容忍多少个epoch没有改进
            min_delta (float): 最小改进阈值
            verbose (bool): 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        if self.best_loss is None:
            self.best_loss = val_metric
            return False
            
        if val_metric > self.best_loss + self.min_delta:
            self.best_loss = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False 