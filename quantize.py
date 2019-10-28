import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            pass
            """
            Apply quantization for the PrunedConv layer.
            --------------Your Code---------------------
            """
            # Changing to cpu to allow use of numpy functions
            og_device = device
            if og_device == 'cuda':
                m.conv = m.conv.cpu()
            m.conv.weight.data[m.mask] = torch.round(m.conv.weight.data[m.mask] * 10**bits) / (10**bits)
            
            # Calculating the kmean clusters
            kmeans = KMeans(n_clusters=2**bits).fit(m.conv.weight.data[m.mask].reshape(-1, 1))
            predictions = kmeans.predict(m.conv.weight.data[m.mask].reshape(-1, 1))
            cluster_centers.append(kmeans.cluster_centers_)
            
            # Applying the clustering
            quantizations = kmeans.cluster_centers_[predictions].reshape(m.conv.weight.data[m.mask].size())
            m.conv.weight.data[m.mask] = torch.from_numpy(quantizations).type(torch.FloatTensor)
            m.conv.weight.data[m.mask] = torch.round(m.conv.weight.data[m.mask] * 10**bits) / (10**bits)
            
            # Returning to original device
            if og_device == 'cuda':
                m.conv = m.conv.to(device)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            --------------Your Code---------------------
            """
            # Changing to cpu to allow use of numpy functions
            og_device = device
            if og_device == 'cuda':
                m.linear = m.linear.cpu()
            m.linear.weight.data[m.mask] = torch.round(m.linear.weight.data[m.mask] * 10**bits) / (10**bits)

            # Calculating the kmean clusters
            kmeans = KMeans(n_clusters=2**bits).fit(m.linear.weight.data[m.mask].reshape(-1, 1))
            predictions = kmeans.predict(m.linear.weight.data[m.mask].reshape(-1, 1))
            cluster_centers.append(kmeans.cluster_centers_)
            
            # Applying the clustering
            quantizations = kmeans.cluster_centers_[predictions].reshape(m.linear.weight.data[m.mask].size())
            m.linear.weight.data[m.mask] = torch.from_numpy(quantizations).type(torch.FloatTensor)
            m.linear.weight.data[m.mask] = torch.round(m.linear.weight.data[m.mask] * 10**bits) / (10**bits)
            
            # Returning to original device
            if og_device == 'cuda':
                m.linear = m.linear.to(device)
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

