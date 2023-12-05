def retrieve(BSBI_instance, query):
    return BSBI_instance.boolean_retrieve(query)

def find_doc_path(BSBI_instance, doc_id):
    BSBI_instance.load()
    return BSBI_instance.doc_id_map[doc_id]

if __name__ == "__main__":
    print(retrieve('batu permata'))
    print(find_doc_path('1368.txt'))