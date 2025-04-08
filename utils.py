import re
import torch
import torch.nn as nn

def grad_vector_to_norm_vector(grad_vector):
    norm_vector = torch.norm(grad_vector, p=2, dim=2)
    norm_vector = norm_vector * (1 / torch.norm(norm_vector, p=float("inf"), dim=1).unsqueeze(1))
    return norm_vector


def inner_product_matrix(grads1, num_of_words):
    num_of_vectors = grads1.shape[0]
    grads1 = grads1[:,:num_of_words,:]
    ret = torch.nn.functional.cosine_similarity(grads1[None, :, :, :], grads1[:, None, :, :], dim=-1, eps=1e-15)
    mask = ~torch.eye(num_of_vectors, dtype=bool)
    ret = ret[mask,...]
    ret = ret.reshape(num_of_vectors * (num_of_vectors-1), -1)
    return ret



def get_topk_words_in_vocab(score_vector, token_splited, num_of_words, vocab, topk):
    tokenmax_val, tokenmax_ind = torch.topk(score_vector, k=num_of_words)
    out_token_scores = []
    count = 0
    for i in range(num_of_words):
        word = token_splited[tokenmax_ind[0][i]]
        value = float(tokenmax_val[0][i])
        if count < topk and word in vocab and value > 0 :
            out_token_scores.append((word, tokenmax_ind[0][i].item(), value))
            count += 1

    return out_token_scores