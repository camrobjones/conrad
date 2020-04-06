from django.shortcuts import render
from django.http import HttpResponse
from django.views import View

def main_home(request):
	return render(request, 'conrad/main_home.html')
	
def about(request):
	return render(request, 'conrad/about.html')
	
class AdsView(View):
    """Replace pub-0000000000000000 with your own publisher ID"""
    def get(self, request, *args, **kwargs):
        line = "google.com, pub-4341380529108117, DIRECT, f08c47fec0942fa0"
        return HttpResponse(line)
