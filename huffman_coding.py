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
    
    # Update frequencies
    for weight in centers:
        if weight[0] in frequency.keys():
            frequency[weight[0]] = frequency.get(weight[0]) + 1
        else:
            frequency[weight[0]] = 1
            
        if weight[0] not in encodings.keys():
            encodings[weight[0]] = []
    
    sorted_freq = sorted(frequency.items() , reverse=False, key=lambda x: x[1])
    huffman_tree[0] = list(sorted_freq[0])
    huffman_tree[0].append(1)
    huffman_tree[1] = list(sorted_freq[1])
    huffman_tree[1].append(0)
    huffman_tree[2] = sorted_freq[0][1] + sorted_freq[1][1] #combined frequency value
    #print(sorted_freq)
    
    # Create Huffman Tree
    i = 2
    pos = 3
    while i < len(sorted_freq):
        node1 = list(sorted_freq[i]) 
        node2 = list(sorted_freq[i + 1])

        # Can add the node to the tree
        if node1[1] >= huffman_tree[pos-1]:
            node1.append(0)
            huffman_tree.append(node1)
            huffman_tree.append( huffman_tree[pos-1] + node1[1] )
            pos += 2
            
            # If there are any nodes to add from previous iteration
            if temp_tree[0] != None:
                huffman_tree.append(temp_tree[0])
                huffman_tree.append(temp_tree[1])
                huffman_tree.append( temp_tree[2] + huffman_tree[pos - 1] )

                temp_tree = [None] * 3
                pos += 3
                
        # Frequency of the node is too small
        else:
            if temp_tree[0] != None:
                huffman_tree.append(temp_tree[0])
                huffman_tree.append(temp_tree[1])
                huffman_tree.append( temp_tree[2] + huffman_tree[pos - 1] )
                pos += 3

            node1.append(1)
            node2.append(0)
            temp_tree[0] = node1
            temp_tree[1] = node2
            temp_tree[2] = node1[1] + node2[1]
            i += 1
                
        # On the final node
        if i + 2 >= len(sorted_freq)and temp_tree[0] != None:
                huffman_tree.append(temp_tree[0])
                huffman_tree.append(temp_tree[1])
                huffman_tree.append( temp_tree[2] + huffman_tree[pos - 1] )
        i += 1
    #print(huffman_tree)
    
    # Set encodings
    for key in encodings.keys():
        found = False
        num_internal = 0
        
        for node in huffman_tree:
            if type(node) is not int and node[0] == key:
                found = True
                encodings[key].append(node[2])
                
            elif found and type(node) is int:
                if num_internal > 0:
                    encodings[key].append(1) #= encodings[key] + (10 ** num_internal)
                num_internal += 1
    #print(encodings)
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