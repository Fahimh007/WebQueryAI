from django.shortcuts import render
from .rag_service import get_rag_answer

def home(request):
    if request.method == 'POST':
        url = request.POST.get('urlInput')
        query = request.POST.get('userQuery')
        if url and query:
            try:
                answer = get_rag_answer(url, query)
                return render(request, 'index.html', {'answer': answer})
            except Exception as e:
                return render(request, 'index.html', {'error': str(e)})
    return render(request, 'index.html', context={})


