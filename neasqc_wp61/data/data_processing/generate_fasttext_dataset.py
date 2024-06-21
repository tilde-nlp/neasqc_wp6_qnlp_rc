import pandas as pd
import fasttext as ft
import fasttext.util as ftu

# Load the dataframe
df = pd.read_csv(
    "~/Downloads/neasqc_data_cvsplit_and_test/ag_news_balanced_test.tsv",
    delimiter="\t",
)

# Rename the "text" column to "sentence"
df.rename(columns={"text": "sentence"}, inplace=True)

ftu.download_model("en", if_exists="ignore")
model = ft.load_model("cc.en.300.bin")
ftu.reduce_model(model, 8)

reduced_embedding_list = []

for sentence in df.sentence.values:
    embedding = model.get_sentence_vector(sentence)
    reduced_embedding_list.append(embedding)

df["reduced_embedding"] = reduced_embedding_list

df.to_csv("./data/datasets/agnews_balanced_test_fasttext.csv", index=False)
