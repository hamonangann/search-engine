import random
import re
import numpy as np
import os
import lightgbm
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class Letor:
    def __init__(self, num_negatives=1, num_latent_topics=200):
        self.num_negatives = num_negatives
        self.num_latent_topics = num_latent_topics
        self.stemmer = PorterStemmer()
        self.stop_words_set = set(StopWordRemoverFactory().get_stop_words())

        self.documents, self.queries, self.q_docs_rel = {}, {}, {}
        self.dataset, self.group_qid_count = [], []
        self.bow_corpus, self.lsi_model, self.ranker = [], None, None
        self.dictionary = Dictionary()

    def _parse_file(self, file_name, delimiter=' ', is_query=False):
        with open(file_name, encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split(delimiter)
                id, content = (int(parts[0]), parts[1:]) if not is_query else (parts[0], parts[1:])
                
                if is_query:
                    self.queries[id] = content 
                else:
                    self.documents[id] = content

    def _create_bow_corpus(self):
        self.bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        self.lsi_model = LsiModel(self.bow_corpus, num_topics=self.num_latent_topics)

    def generate_whole_dataset(self):
        self._parse_file(os.path.join("retriever", "qrels-folder", "train_docs.txt"))
        self._create_bow_corpus()
        self._parse_file((os.path.join("retriever", "qrels-folder", "train_queries.txt")), is_query=True)
        self.q_docs_rel = self._parse_qrels(self.queries, (os.path.join("retriever", "qrels-folder", "train_qrels.txt")))

        self.dataset, self.group_qid_count = self._make_dataset(self.queries, self.q_docs_rel)

    def _make_dataset(self, queries, q_docs_rel):
        dataset, group_qid_count = [], []
        all_docs = list(self.documents.values())
        for q_id, docs_rels in q_docs_rel.items():
            group_qid_count.append(len(docs_rels) + self.num_negatives)
            dataset.extend([(queries[q_id], self.documents[doc_id], rel) for doc_id, rel in docs_rels])
            dataset.extend([(queries[q_id], random.choice(all_docs), 0) for _ in range(self.num_negatives)])
        return dataset, group_qid_count

    def _parse_qrels(self, queries, file_name):
        q_docs_rel = {}
        with open(file_name, encoding='utf-8') as file:
            for line in file:
                q_id, doc_id, rel = line.strip().split(' ')
                if q_id in queries and int(doc_id) in self.documents:
                    q_docs_rel.setdefault(q_id, []).append((int(doc_id), int(rel)))
        return q_docs_rel

    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        if len(rep) == self.num_latent_topics:
            return rep
        return [0.] * self.num_latent_topics

    def generate_validation_set(self):
        self._parse_file((os.path.join("retriever", "qrels-folder", "val_queries.txt")), is_query=True)
        qrels = self._parse_qrels(self.queries, (os.path.join("retriever", "qrels-folder", "val_qrels.txt")))
        return self._make_dataset(self.queries, qrels)
    
    def _feature_extraction(self, query, doc):
        v_q, v_d = self.vector_rep(query), self.vector_rep(doc)
        q_set, d_set = set(query), set(doc)
        return v_q + v_d + [
            len(q_set & d_set) / len(q_set | d_set),  # Jaccard similarity
            cosine(v_q, v_d),  # Cosine distance
            len(q_set & d_set),  # Common words count
            sum(len(word) for word in q_set) / (len(q_set) or 1),  # Avg. word length in query
            sum(len(word) for word in d_set) / (len(d_set) or 1)  # Avg. word length in doc
        ]

    def split_data(self, dataset):
        X = [self._feature_extraction(query, doc) for query, doc, _ in dataset]
        Y = [rel for _, _, rel in dataset]
        return np.array(X), np.array(Y)
    
    def get_document_contents(self, file_name):
        document_contents = []
        with open(os.path.join("retriever", "collections", file_name), 'rb') as file:
            try:
                document = file.read().decode().lower()
            except UnicodeDecodeError:
                document = file.read().decode('latin-1').lower()
            tokenized_document = re.findall(r'\w+', document)
            document_contents = document_contents + self.preprocess_text(" ".join(tokenized_document))
        return " ".join(document_contents)
    
    def preprocess_text(self, text):
        result = []
        tokenized_document = re.findall(r'\w+', text)
        for term in tokenized_document:
            if term not in self.stop_words_set:
                result.append(self.stemmer.stem(term.lower()))
        return result
    
    def train(self):
        X, Y = self.split_data(self.dataset)

        self.ranker = lightgbm.LGBMRanker(
            objective="lambdarank", boosting_type="gbdt", n_estimators=150,
            importance_type="gain", metric="ndcg", num_leaves=70, learning_rate=0.04,
            max_depth=7, random_state=2023
        )
        self.ranker.fit(X, Y, group=self.group_qid_count)

    def rerank(self, query, docs):
        X_unseen = [self._feature_extraction(query.split(), doc.split()) for _, doc in docs]
        scores = self.ranker.predict(np.array(X_unseen))
        return sorted(zip(scores, [did for (did, _) in docs]), key=lambda tup: tup[0], reverse=True)

    def initialize_model(self):
        self.generate_whole_dataset()
        self.train()

if __name__ == '__main__':
    letor = Letor()
    letor.initialize_model()