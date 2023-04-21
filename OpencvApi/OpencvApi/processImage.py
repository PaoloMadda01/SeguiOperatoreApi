from django.http import HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        # process the images and return an integer response
        ...
    else:
        return HttpResponseNotAllowed(['POST'])

