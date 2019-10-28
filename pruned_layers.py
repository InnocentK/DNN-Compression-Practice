import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out
        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        total_weights = 0
        pruned_weights = 0
        
        # Changing to cpu to allow use of numpy functions
        og_device = device
        if og_device == 'cuda':
            self.linear = self.linear.cpu()
        
        # Calculating threshold value
        weights = torch.abs(self.linear.weight.data)
        thresh = np.percentile(weights, q)
        
        # Updating weights
        self.mask = torch.abs(self.linear.weight.data) < thresh
        mask = torch.ones(self.mask.size()) - self.mask.float()
        self.linear.weight.data *= mask
        pruned_weights += self.mask.numel()
        total_weights += self.linear.weight.data.numel()
                
        # Calculating and storing the sparsity
        self.sparsity = pruned_weights / total_weights
        self.mask = torch.abs(self.linear.weight.data) >= thresh
        
        # Returning to original device
        if og_device == 'cuda':
            self.linear = self.linear.to(device)
            
        pass


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        total_weights = 0
        pruned_weights = 0
        
        # Changing to cpu to allow use of numpy functions
        og_device = device
        if og_device == 'cuda':
            self.linear = self.linear.cpu()
        
        # Calculating threshold value
        weights = torch.abs(self.linear.weight.data)
        thresh = weights.std() * s
        
        # Updating weights
        self.mask = torch.abs(self.linear.weight.data) < thresh
        mask = torch.ones(self.mask.size()) - self.mask.float()
        self.linear.weight.data *= mask
        pruned_weights += self.mask.numel()
        total_weights += self.linear.weight.data.numel()
                
        # Calculating and storing the sparsity
        self.sparsity = pruned_weights / total_weights
        self.mask = torch.abs(self.linear.weight.data) >= thresh
        
        # Returning to original device
        if og_device == 'cuda':
            self.linear = self.linear.to(device)
        
        pass

class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Expand and Transpose to match the dimension
        self.mask = np.ones_like([out_channels, in_channels, kernel_size, kernel_size])

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        total_weights = 0
        pruned_weights = 0
        
        # Changing to cpu to allow use of numpy functions
        og_device = device
        if og_device == 'cuda':
            self.conv = self.conv.cpu()
        
        # Calculating threshold value
        weights = torch.abs(self.conv.weight.data)
        thresh = np.percentile(weights, q)
        
        # Updating weights
        self.mask = torch.abs(self.conv.weight.data) < thresh
        mask = torch.ones(self.mask.size()) - self.mask.float()
        self.conv.weight.data *= mask
        pruned_weights += self.mask.numel()
        total_weights += self.conv.weight.data.numel()
                
        # Calculating and storing the sparsity
        self.sparsity = pruned_weights / total_weights
        self.mask = torch.abs(self.conv.weight.data) >= thresh
        
        # Returning to original device
        if og_device == 'cuda':
            self.conv = self.conv.to(device)

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        total_weights = 0
        pruned_weights = 0
        
        # Changing to cpu to allow use of numpy functions
        og_device = device
        if og_device == 'cuda':
            self.conv = self.conv.cpu()
        
        # Calculating threshold value
        weights = torch.abs(self.conv.weight.data)
        thresh = weights.std() * s
        
        # Updating weights
        self.mask = torch.abs(self.conv.weight.data) < thresh
        mask = torch.ones(self.mask.size()) - self.mask.float()
        self.conv.weight.data *= mask
        pruned_weights += self.mask.numel()
        total_weights += self.conv.weight.data.numel()
                
        # Calculating and storing the sparsity
        self.sparsity = pruned_weights / total_weights
        self.mask = torch.abs(self.conv.weight.data) >= thresh
        
        # Returning to original device
        if og_device == 'cuda':
            self.conv = self.conv.to(device)
        
        pass

