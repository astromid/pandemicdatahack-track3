from gensim.models import fasttext
import pandas as pd
from tqdm import tqdm_notebook
import re
import numpy as np
import pickle

model = fasttext.load_facebook_model("../data/ftdp/ft_native.100.ru_wiki_lenta_lower_case.bin")
emlp = pd.read_csv("../data/raw/employements_mult_new.csv", sep=";")

text_cols = ["employer", "achievements", "responsibilities", "position_clean"]

new_fts = {}
for col in text_cols:
    def prepare(text):
        text = text.lower()
        tokens = re.findall("[\w']+", text)
        return " ".join(tokens)
    emlp[col] = emlp[col].fillna("").apply(prepare)
#     vectorizer = TfidfVectorizer()
#     vectorizer.fit(emlp[col])
#     idf = dict(zip(vectorizer.idf_, vectorizer.get_feature_names()))
    def get_vector_by_text(text):
        tokens = text.split(" ")
        return np.mean([model.wv.get_vector(token) for token in tokens if token in model.wv], axis=0)
    
    vecs = []
    for _, a in tqdm_notebook(emlp[col].iteritems(), total=len(emlp)):
        vecs.append(get_vector_by_text(a))
    new_fts[col] = pd.DataFrame(np.array(vecs), columns=[f"{col}_{i}" for i in range(100)])

res2 = pd.concat([emlp[["id"]], *list(new_fts.values())], axis=1)

pickle.dump(res2, open("../data/employements_mult_new_ft_1.pkl", "wb"))

