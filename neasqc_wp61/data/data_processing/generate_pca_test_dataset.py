import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv("../datasets/agnews_balanced_test_bert.csv")

# Convert the string representation to actual lists of embeddings
df["sentence_embedding"] = df["sentence_embedding"].apply(eval)

embeddings = df["sentence_embedding"].tolist()
embeddings_array = np.array(embeddings)

# Fit PCA on embeddings
pca = PCA(n_components=8)
reduced_embeddings = pca.fit_transform(embeddings_array)

df["reduced_embedding"] = reduced_embeddings.tolist()
df.to_csv("../datasets/agnews_balanced_test_bert_pca.csv", index=False)
