import numpy as np
import torch 
from torch import nn
from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, CharTensor, LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, CharTensor, LongTensor

from transformers import BertTokenizer
from sklearn import metrics

from sklearn.metrics import classification_report
from torch.nn.functional import one_hot, binary_cross_entropy


def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    '''
        Args: 
            q_seqs: the question(KC) sequence with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequence with the size of \
                [batch_size, some_sequence_length]

            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs
            
            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
    '''

    proc_q_seqs = []
    proc_r_seqs = []

    # seq_len means the sequence length to match q_seqs and r_seqs to the same length.
    # q_seq is a list holding the index list for the user's skill.
    # You can think of it as cutting the given q, r sequences by seq_len
    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq): # While i + seq_len + 1 is less than the length of the given question set, e.g.) 0 + 100 + 1 < 128
            proc_q_seqs.append(q_seq[i:i + seq_len + 1]) # Add elements from i to i + seq_len + 1 range to the question sequence e.g.) Assign the array sequence of elements from 0 to 0 + 100 + 1 to proc_q
            proc_r_seqs.append(r_seq[i:i + seq_len + 1]) # Same as above. e.g.) Assign the array sequence of elements from 0 to 0 + 100 + 1 to proc_r

            i += seq_len + 1 # Increase i by seq_len + 1 to make it larger than len(q_seq)

        # Concatenate the sequences cut by seq_len with padding values
        # Since the sequences are shorter, replace the remaining part with padding values; otherwise, append the original sequence with padding values
        proc_q_seqs.append(
            np.concatenate([
                q_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq))) # Create elements of an array containing the padding value (here 0, assuming q_seq is 128, it will create 1 element as 129 - 128)
            ]) 
        )
        proc_r_seqs.append(
            np.concatenate([
                r_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            ])
        )
        # The last one element is padded and added

    return proc_q_seqs, proc_r_seqs


def collate_fn(batch, pad_val=-1):
    '''
    This function for torch.utils.data.DataLoader

    Returns:
        q_seqs: the question(KC) sequences with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        r_seqs: the response sequences with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        qshft_seqs: the question(KC) sequences which were shifted \
            one step to the right with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        rshft_seqs: the response sequences which were shifted \
            one step to the right with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        mask_seqs: the mask sequences indicating where \
            the padded entry is with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
    '''

    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []


    # q_seq and r_seq are taken until the second to last element (the last is the padding value)
    # q_shft and rshft take elements starting from the first (since it's a right-shifted value)
    for q_seq, r_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1])) 
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))

    # pad_sequence, the first argument is sequence, the second is to bring batch_size as the first argument, and the third argument is the padding value
    # Padding is done based on the longest sequence within the sequences; the unmatched parts are filled with padding_value
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )


    # Generate masking sequences 
    # General question sequences: all values different from the padding value are treated as 1, padding values are treated as 0.
    # General question padding sequences: shifted sequence values different from the padding value are treated as 1, padding values are treated as 0.
    # Masking sequences: all padding sequence values are 0, both values not being padding are treated as 1. (original and shift sequences' values)
    # For example, if both the current and the next value are not padding values, it is treated as 1; if either is padding, it is treated as 0.
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    # Even if the next value (shift value) is padding, the value becomes 0 by the masking sequence. Otherwise, it holds the original sequence data.
    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs
    

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs
