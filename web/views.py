from django.shortcuts import render
from retriever.search import retrieve, find_doc_path

import time

# Create your views here.
def index(request):
    return render(request, 'index.html')

def list_docs(request):
    time_start = time.time()

    query = request.GET.get('query')
    doc_names, doc_ids = retrieve(query)

    docs = []
    for i in range(len(doc_names)):
        docs.append({'name': doc_names[i], 'id': doc_ids[i]})

    time_stop = time.time()
    time_count = time_stop - time_start

    return render(request, 'list_docs.html', {'query': query, 'docs': docs, 'time_count': time_count})

def view_doc(request, doc_id):
    doc_path = find_doc_path(int(doc_id))
    doc_name = doc_path.split("/")[-1]
    with open(doc_path, 'r') as f:
        return render(request, 'view_doc.html', {'doc_id': doc_id, 'doc_name': doc_name, 'doc_content': f.read()})