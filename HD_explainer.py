from utils import inner_product_matrix, grad_vector_to_norm_vector, get_topk_words_in_vocab
from typing import List
from view_utils import show_in_notebook
from classifier import Classifier
import torch
import matplotlib.pyplot as plt
import traceback

class HD_Explainer:

    def __init__(self, model: Classifier, surrogate_models: List[Classifier], token_split, vocab, max_len):
        self.model_to_explain = model
        self.surrogate_models = surrogate_models
        self.token_split = token_split
        self.vocab_dict = vocab
        self.max_len = max_len

    def get_all_grads(self, x, label):
        token_splited = self.token_split(x)[0]
        num_of_words = min(len(token_splited),self.max_len)

        with torch.no_grad():
            prediction = self.model_to_explain.classify(x)
            assert prediction == label, "the data is not correctly labeled, incorrect labels cannot be explained"
            model_grad = self.model_to_explain.get_gradient(x, 1-label)[0,:num_of_words,:]
            all_predictions = torch.tensor([m.classify(x) for m in self.surrogate_models])
            assert (all_predictions==label).sum() > 2, "the data is classify incorrectly by too many surrogate classifiers"
            other_grads = [self.surrogate_models[i].get_gradient(x, 1-label)[0,:num_of_words,:] for i in range(len(self.surrogate_models)) if all_predictions[i]==label]

        all_grads = other_grads+[model_grad]
        return all_grads


    def explain(self, x, label):
        string_input = x[0]
        token_splited = self.token_split(x)[0]
        num_of_words = min(len(token_splited),self.max_len)

        all_grads = self.get_all_grads(x,label)
        tensor_grads = torch.stack(all_grads, dim=0)
        all_inners_product = inner_product_matrix(tensor_grads, num_of_words)
        norms = grad_vector_to_norm_vector(tensor_grads)
        all_inners_product = abs(all_inners_product)

        avg_inner_product = torch.mean(all_inners_product, dim=0)
        avg_norm = torch.mean(norms, dim=0)
        norm_treshold = avg_norm.median()

        avg_inner_product_small_norm = avg_inner_product.clone()
        avg_inner_product_small_norm[avg_norm>norm_treshold] = 0 

        small_norm = avg_norm.clone()
        small_norm[avg_norm>norm_treshold] = 0

        topk_abs_inner = get_topk_words_in_vocab(avg_inner_product.unsqueeze(0), token_splited, num_of_words, self.vocab_dict, topk=15)
        print("top correlated angle : ", topk_abs_inner) 
        topk_abs_inner_smlnorm = get_topk_words_in_vocab(avg_inner_product_small_norm.unsqueeze(0), token_splited, num_of_words, self.vocab_dict, topk=15)
        print("top correlated angle small norm: ", topk_abs_inner_smlnorm) 
        topk_abs_norm = get_topk_words_in_vocab(avg_norm.unsqueeze(0), token_splited, num_of_words, self.vocab_dict, topk=15)
        print("top gradient norm : ", topk_abs_norm) 
        topk_abs_norm_small_norm = get_topk_words_in_vocab(small_norm.unsqueeze(0), token_splited, num_of_words, self.vocab_dict, topk=15)
        print("top gradient norm small norm: ", topk_abs_norm_small_norm) 
        try:
            show_in_notebook(string_input, topk_abs_inner_smlnorm, self.token_split)
            show_in_notebook(string_input, topk_abs_norm, self.token_split)

        except Exception as e:
            print("Exception", e)
            traceback.print_exc() 

    def plot(self, x, label):
        token_splited = self.token_split(x)[0]
        num_of_words = min(len(token_splited),self.max_len)
        all_grads = self.get_all_grads(x,label)

        tensor_grads = torch.stack(all_grads, dim=0)
        all_inners_product = inner_product_matrix(tensor_grads, num_of_words)
        norms = grad_vector_to_norm_vector(tensor_grads)
        all_inners_product = abs(all_inners_product)


        max_inner_product, _ = torch.max(all_inners_product, dim=0)
        min_inner_product, _ = torch.min(all_inners_product, dim=0)
        avg_inner_product = torch.mean(all_inners_product, dim=0)
        max_inner_product_error = max_inner_product - avg_inner_product
        min_inner_product_error = avg_inner_product - min_inner_product


        max_norms, _ = torch.max(norms, dim=0)
        min_norms, _ = torch.min(norms, dim=0)
        avg_norm = torch.mean(norms, dim=0)
        print("avg_norm", avg_norm)

        max_norms_error = max_norms - avg_norm
        min_norms_error = avg_norm - min_norms

        norm_treshold = avg_norm.median()

        avg_inner_product_small_norm = avg_inner_product.clone()
        avg_inner_product_small_norm[avg_norm>norm_treshold] = 0 

        small_norm = avg_norm.clone()
        small_norm[avg_norm>norm_treshold] = 0

        topk_abs_inner_smlnorm = get_topk_words_in_vocab(avg_inner_product_small_norm.unsqueeze(0), token_splited, num_of_words, self.vocab_dict, topk=15)
        
        plt.rcParams['figure.figsize'] = [20, 20]
        plt.errorbar(avg_inner_product, avg_norm, xerr=[min_inner_product_error.tolist(), max_inner_product_error.tolist()],
                    yerr=[min_norms_error.tolist(), max_norms_error.tolist()], fmt='ko', elinewidth=0.1)


        plt.xlabel("inner product per word")
        plt.ylabel("gradient norm per word")
        for i in range(num_of_words):
            if token_splited[i] in self.vocab_dict:
                c = "blue"
                font_size=14
                topk_abs_inner_smlnorm_inds = [ind for (word,ind,score) in topk_abs_inner_smlnorm]
                if i in topk_abs_inner_smlnorm_inds:
                    c="green"
                    font_size = 20
                plt.annotate(token_splited[i], (avg_inner_product.cpu()[i], avg_norm[i]), color=c, fontsize=font_size)
            else:
                c = "red"
        
        plt.axvline(x = 1/torch.sqrt(torch.tensor(all_grads[0].nelement())/num_of_words), color = 'r', label = '')
        plt.axvline(x = -1/torch.sqrt(torch.tensor(all_grads[0].nelement())/num_of_words), color = 'r', label = '')
        plt.title("score vector distribution")
        plt.show()

        print("small norm only")
        plt.rcParams['figure.figsize'] = [20, 50]
        plt.errorbar(avg_inner_product_small_norm[avg_norm<norm_treshold], avg_norm[avg_norm<norm_treshold], xerr=[min_inner_product_error[avg_norm<norm_treshold].tolist(), max_inner_product_error[avg_norm<norm_treshold].tolist()],
                    yerr=[min_norms_error[avg_norm<norm_treshold].tolist(), max_norms_error[avg_norm<norm_treshold].tolist()], fmt='ko', elinewidth=0.1)

        plt.yticks(torch.arange(0, norm_treshold, step=0.001))
        plt.ylim(0,norm_treshold)
        plt.xlabel("inner product per word")
        plt.ylabel("gradient norm per word")
        for i in range(num_of_words):
            if token_splited[i] in self.vocab_dict and avg_norm[i]<norm_treshold:
                c = "blue"
                font_size=14
                topk_abs_inner_smlnorm_inds = [ind for (word,ind,score) in topk_abs_inner_smlnorm]
                if i in topk_abs_inner_smlnorm_inds:
                    c="green"
                    font_size = 20
                plt.annotate(token_splited[i], (avg_inner_product.cpu()[i], avg_norm[i]), color=c, fontsize=font_size)
            else:
                c = "red"
        
        plt.axvline(x = 1/torch.sqrt(torch.tensor(all_grads[0].nelement())/num_of_words), color = 'r', label = '')
        plt.axvline(x = -1/torch.sqrt(torch.tensor(all_grads[0].nelement())/num_of_words), color = 'r', label = '')
        plt.title("score vector distribution- small norm only")
        plt.show()
        return True
