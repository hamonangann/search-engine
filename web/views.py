from django.shortcuts import render
from retriever.search import retrieve

# Create your views here.
def index(request):
    return render(request, 'index.html')

def list_docs(request):
    query = request.GET.get('query')
    return render(request, 'list_docs.html', {'docs': retrieve(query)})

def view_doc(request, doc_id):
    # doc = Doc.objects.get(id=doc_id)
    with open('retriever/collections/0/' + id)
    return render(request, 'view_doc.html', {'doc': None})