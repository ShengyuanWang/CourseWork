from django.shortcuts import render, HttpResponse
from app01.models import  Department,UserInfo
# Create your views here.

def index(request):
    return HttpResponse("Welcome")

def user_list(request):
    return render(request,"user_list.html")

def user_add(request):
    return render(request, "user_add.html")

def orm(request):
    # test orm
    # Department.objects.create(title="销售部")
    # Department.objects.create(title="IT部")
    # Department.objects.create(title="运营部")
    UserInfo.objects.create(name="Alan", password="123", age=21)
    return HttpResponse("Success")
