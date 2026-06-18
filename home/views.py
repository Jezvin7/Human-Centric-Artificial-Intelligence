# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Jesvin Varghese", "matriculation": "674425"},
        {"name": "Shashank Mallappa Gujjal", "matriculation": "680181"},
    ]
    
    projects = [
        {"name": "Project 1", "url_name": "project1:index"},
        {"name": "Project 2", "url_name": "project2:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))


    