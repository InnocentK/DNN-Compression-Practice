import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn
import heapq

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding for each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    """
    """
    Generate Huffman Coding and Frequency Map according to incoming weights and centers (KMeans centriods).
    --------------Your Code---------------------
    """
    encodings = {}
    frequency = {}
    huffman_tree = [None] * 3
    temp_tree = [None] * 3
    pos = 3
    
    # Update frequencies
    #print(centers)
    for weight in centers:
        if weight[0] in frequency.keys():
            frequency[weight[0]] = frequency.get(weight[0]) + 1
        else:
            frequency[weight[0]] = 1
            
        #if weight[0] not in encodings.weight():
            #encodings[weight[0]] = 0
    
    sorted_freq = sorted(frequency.items() , reverse=False, key=lambda x: x[1])
    huffman_tree[0] = sorted_freq[0]
    huffman_tree[1] = sorted_freq[1]
    huffman_tree[2] = sorted_freq[0][1] + sorted_freq[1][1] #combined frequency value
    print(sorted_freq)
    
    i = 2
    #for i in range(3,len(sorted_freq),1):
    while i < len(sorted_freq):
        print(huffman_tree)
        node1 = sorted_freq[i] 
        node2 = sorted_freq[i + 1]
        #print(node1[1] >= huffman_tree[pos-1], node1[1], huffman_tree[pos-1])
        print("pos", pos, "i", i)

        if node1[1] >= huffman_tree[pos-1] and temp_tree[0] == None:
            huffman_tree.append(node1)
            huffman_tree.append( huffman_tree[pos-1] + node1[1] )
            pos += 2
            
        elif node1[1] >= huffman_tree[pos-1]:
            huffman_tree.append(node1)
            huffman_tree.append( huffman_tree[pos-1] + node1[1] )
            huffman_tree.append(temp_tree[0])
            huffman_tree.append(temp_tree[1])
            huffman_tree.append( temp_tree[2] + huffman_tree[pos + 1] )

            temp_tree = [None] * 3
            pos += 5
            
        elif temp_tree[0] != None:
            huffman_tree.append(temp_tree[0])
            huffman_tree.append(temp_tree[1])
            huffman_tree.append( temp_tree[2] + huffman_tree[pos - 1] )

            temp_tree[0] = node1
            temp_tree[1] = node2
            temp_tree[2] = node1[1] + node2[1]
            pos += 3
            i += 1
            
        else:
            temp_tree[0] = node1
            temp_tree[1] = node2
            temp_tree[2] = node1[1] + node2[1]
            i += 1
                
        # On the final node
        print(temp_tree, i)
        if i + 2 >= len(sorted_freq)and temp_tree[0] != None:
                huffman_tree.append(temp_tree[0])
                huffman_tree.append(temp_tree[1])
                huffman_tree.append( temp_tree[2] + huffman_tree[pos - 1] )
        i += 1
    print(huffman_tree)
    
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map