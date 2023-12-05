from .bsbi import BSBIIndex
from .compression import VBEPostings

def retrieve(query):
    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(data_path = 'retriever/collections/', \
                            postings_encoding = VBEPostings, \
                            output_path = 'retriever/index')

    # queries = ["pupil mata", "aktor", "batu permata"]
    return BSBI_instance.boolean_retrieve(query)

def find_doc_path(doc_id):
    # sebelumnya sudah dilakukan indexing
    # BSBIIndex hanya sebagai abstraksi untuk index tersebut
    BSBI_instance = BSBIIndex(data_path = 'retriever/collections/', \
                            postings_encoding = VBEPostings, \
                            output_path = 'retriever/index')
    BSBI_instance.load()

    return BSBI_instance.doc_id_map[doc_id]

if __name__ == "__main__":
    print(retrieve('batu permata'))
    print(find_doc_path('1368.txt'))