from django.shortcuts import render

def main_home(request):
	return render(request, 'conrad/main_home.html')
	
def about(request):
	return render(request, 'conrad/about.html')
	