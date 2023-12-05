from django.apps import AppConfig

from retriever.bsbi import BSBIIndex
from retriever.cache import DocCache
from retriever.compression import VBEPostings

class WebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'web'

    def __init__(self, app_name, app_module):
        super(WebConfig, self).__init__(app_name, app_module)
        self.bsbi_instance = None
        self.cache = None

    def ready(self):
        # initialize retriever instance
        self.bsbi_instance = BSBIIndex(data_path = 'retriever/collections/', \
                                postings_encoding = VBEPostings, \
                                output_path = 'retriever/index')

        # initialize cache
        self.cache = DocCache()
        for i in range(len(self.bsbi_instance.doc_id_map)):
            doc_path = self.bsbi_instance.doc_id_map[i]
            self.cache.set(i, doc_path)

