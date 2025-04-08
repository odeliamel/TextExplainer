import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import decomposition

# import lime
# from lime import lime_text
# from lime.lime_text import LimeTextExplainer
import sys, os
sys.path.insert(0, os.path.abspath('../..'))
from exai.explainers.lime_explainer import LimeExplainer
from exai.explainers.TE_lime_explainer import TELimeExplainer
from exai.explainers.shap_explainer import ShapExplainer
from exai.explainers.gradient_explainer import GradientExplainer
from exai.classifier import Classifier
from exai.dataset import TestData
from exai.different_init.grads_to_score import analyze

# %%
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


# %%
dataset_path="/home/azureuser/cloudfiles/code/Users/t-omelamed/exai/randomLabel/data/data_partition_*.csv"
dataset = TestData(test_data_path=dataset_path)
classifier = Classifier("org")

input_dim = 1000*96

def preproccess():
    data_as_matrix = torch.tensor([])
    # itera = dataset.get_data_iterator(batch_size=5000)
    # for itera in [dataset.get_data_iterator(batch_size=5000),dataset.get_malware_data_iterator(batch_size=5000)]:
    for itera in [dataset.get_balanced_data_iterator(batch_size=5000)]:
        for data, target in itera:
            print("pca data batch", len(data))
            chrs, tkns = classifier.preprocess_tokenizer(data)
            ppinput = classifier.model.embbeding_forward(chrs,tkns)
            # emb_input_chrs = Variable(ppinput[0])
            # emb_input_tokens = Variable(ppinput[1])

            data = ppinput[1].detach().clone().cpu()
            print("data shape", data.shape)

            data_as_matrix = torch.cat([data_as_matrix, data.reshape(data.shape[0], input_dim)], dim=0)
            print("data matrix shape", data_as_matrix.shape)
            if data_as_matrix.shape[0] >= 5000:
                break

        no_rows, no_columns = data_as_matrix.size()
        row_means = torch.mean(data_as_matrix, 1).unsqueeze(1)
        #Expand the matrix in order to have the same shape as X and substract, to center
        for_subtraction = row_means.expand(no_rows, no_columns)
        X = data_as_matrix - for_subtraction #centered

        pca = decomposition.PCA()
        # PCA for dimensionality redcution (non-visualization)
        pca.n_components = min(input_dim, data_as_matrix.shape[0])
        pca_data = pca.fit_transform(X)

        print("done pca")
        percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
        cum_var_explained = np.cumsum(percentage_var_explained)
        print(cum_var_explained)
        # Plot the PCA spectrum
        plt.figure(1, figsize=(6, 4))
        plt.clf()
        plt.plot(cum_var_explained, linewidth=2, label="Cumulative Variance")
        plt.axhline(y=0.95, color='r', linestyle='-', label="95%")
        plt.axis('tight')
        plt.grid()
        plt.xlabel('# Features')
        plt.legend()
        # plt.ylabel('Cumulative Explained Variance')
        plt.show()

preproccess()