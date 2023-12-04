from django.shortcuts import render
from retriever.search import retrieve
import time

# Create your views here.
def index(request):
    return render(request, 'index.html')

def list_docs(request):
    time_start = time.time()

    query = request.GET.get('query')
    docs = retrieve(query)

    time_stop = time.time()
    time_count = time_stop - time_start

    return render(request, 'list_docs.html', {'query': query, 'docs': docs, 'time_count': time_count})

def view_doc(request, doc_id):
    with open('retriever/collections/0/' + doc_id, 'r') as f:
        return render(request, 'view_doc.html', {'doc_name': doc_id, 'doc_content': f.read()})