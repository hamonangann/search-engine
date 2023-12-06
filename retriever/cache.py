import os

# perform in-memory cache doc to boost SERP load
class DocCache:
    def __init__(self) -> None:
        self.doc_summary = {}
    
    def set(self, doc_id, doc_path):
        try:
            with open(os.path.join('retriever', 'collections', doc_path), 'r') as f:
                self.doc_summary[doc_id] = f.read(300)
                return self.doc_summary[doc_id]
        except:
            return None
        
    def get(self, doc_id):
        try:
            return self.doc_summary[doc_id]
        except:
            return None