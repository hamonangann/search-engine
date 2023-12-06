from retriever.bsbi import BSBIIndex
from retriever.compression import VBEPostings
from retriever.letor import Letor

import os

def retrieve(BSBI_instance, letor_instance, query):
    k1_letor = 1.2
    b_letor = 0.75
    search_results = BSBI_instance.retrieve_bm25_wand(query, k=100, k1=k1_letor, b=b_letor)

    if len(search_results) <= 0:
        return []
    
    docs = []
    for (_, doc_path, doc_id) in search_results:
        doc_text = letor_instance.get_document_contents(doc_path)
        doc_representation = (doc_id, doc_text)
        docs.append(doc_representation)

    reranking_result = letor_instance.rerank(query, docs)

    result_with_path = []

    for (score, doc_id) in reranking_result:
        doc_path = BSBI_instance.doc_id_map[doc_id]
        result_with_path.append((score, doc_path, doc_id))

    return result_with_path

def find_doc_path(BSBI_instance, doc_id):
    BSBI_instance.load()
    return BSBI_instance.doc_id_map[doc_id]

if __name__ == "__main__":
    bsbi_instance = BSBIIndex(data_dir=os.path.join("retriever", "collections"),
                              postings_encoding=VBEPostings,
                              output_dir=os.path.join("retriever", "index"))
    bsbi_instance.load()

    letor_instance = Letor()
    letor_instance.initialize_model()

    print(retrieve(bsbi_instance, letor_instance, 'batu permata'))
    print("doc path", find_doc_path(bsbi_instance, 1))