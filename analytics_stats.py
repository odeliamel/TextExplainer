import torch
from utils import grad_vector_to_norm_vector
import matplotlib.pyplot as plt


# creates histogram of the norms of the gradient of different words, demonstrating the high-dimensional phenomenon of small vs. big norms
# on manifold perturbation will be larger, meaning smaller gradient norms.
def gradient_norm_stats(models, dataset_iterator, token_split=lambda x: [text.split() for text in x], max_len=500):
    all_norms = []
    for input, label in dataset_iterator:
    # print(i)
        inp = {"features":input, "label":1-label}
        token_splited = token_split(inp)[0]
        num_of_words = min(len(token_splited), max_len)
        all_predictions = torch.tensor([m.classify(inp) for m in models])

        grads = [models[i].get_gradient(inp, 1-label)[0,:num_of_words,:] for i in range(len(models)) if all_predictions[i]==label]
        [models[i].model.zero_grad() for i in range(len(models))]
        # print("grad shape", grad.shape)
        grads = grads.detach().cpu()
        norms = grad_vector_to_norm_vector(grads)
        all_norms.append(norms)

    all_norms = torch.flatten(torch.tensor(all_norms))
    max_norm = round(torch.max(all_norms))
    bins = range(0, max_norm, round(max_norm/20))
    plt.hist(all_norms, bins, histtype='bar', rwidth=0.8)
    plt.xlabel('Word norm')
    plt.ylabel('Amount')
    plt.title('how does the norms look like')
    plt.legend()
    plt.show()


def plot(grad1, grad2, inners):
        y = torch.ones_like(grad1[0]).cpu()
        plt.plot(grad1[0].cpu(), y-1, 'og')
        plt.plot(grad2[0].cpu(), y, 'or')
        plt.title("grad coordinate. green for true labels")
        plt.show()

        true_model_grad_norm = torch.norm(grad1, p=2,dim=2)
        flipped_model_grad_norm = torch.norm(grad2, p=2,dim=2)
        y = torch.ones_like(true_model_grad_norm[0]).cpu()
        plt.plot(true_model_grad_norm[0].cpu(), y-1, 'og')
        plt.plot(flipped_model_grad_norm[0].cpu(), y, 'or')
        plt.title("norm grad coordinate. green for true labels")
        plt.show()

        plt.plot(inners , y-1, '+')
        plt.plot([-0.125,0.125] , [0,0], 'ro')  
        plt.title("inner product between cooralation vectors")
        plt.show()