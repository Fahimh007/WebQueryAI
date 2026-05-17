from django.shortcuts import render
from .rag_service import ask_website

def home(request):

    context = {}

    if request.method == "POST":

        url = request.POST.get("urlInput")
        query = request.POST.get("userQuery")

        if not url or not query:
            context["error"] = "URL and query are required."
            return render(request, "index.html", context)

        try:
            result = ask_website(url, query)

            context["answer"] = result["answer"]
            context["sources"] = result["sources"]
            context["url"] = url
            context["query"] = query

        except Exception as e:
            context["error"] = str(e)

    return render(request, "index.html", context)


