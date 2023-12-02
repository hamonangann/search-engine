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

if __name__ == "__main__":
    print(retrieve('batu permata'))