from django.apps import apps
from django.shortcuts import render
from retriever.search import retrieve, find_doc_path
from .apps import WebConfig

import time

# Create your views here.
def index(request):
    return render(request, 'index.html')

def list_docs(request):
    time_start = time.time()

    my_app_config = apps.get_app_config('web')

    query = request.GET.get('query')
    print(query)
    if query == None:
        query = ""

    page = request.GET.get('page')
    if page == None:
        page = 1
    else:
        page = int(page)

    doc_names, doc_ids = retrieve(my_app_config.bsbi_instance, query)
    docs = []
    for i in range((page-1)*5, min(len(doc_ids), page*5)):
        doc_summary = my_app_config.cache.get(doc_ids[i])

        if doc_summary == None:
            doc_summary = my_app_config.cache.set(doc_ids[i], doc_names[i])

        docs.append({'name': doc_names[i], 'id': doc_ids[i], 'summary': doc_summary})

    time_stop = time.time()
    time_count = time_stop - time_start

    return render(request, 'list_docs.html', {'page': page, 'query': query, 'docs': docs, 'time_count': time_count})

def view_doc(request, doc_id):
    my_app_config = apps.get_app_config('web')

    doc_path = find_doc_path(my_app_config.bsbi_instance, int(doc_id))

    with open(doc_path, 'r') as f:
        return render(request, 'view_doc.html', {'doc_id': doc_id, 'doc_name': doc_path, 'doc_content': f.read()})