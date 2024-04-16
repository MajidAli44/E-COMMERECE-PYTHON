from django.conf import settings

def api_base_url(request):
    return {'SITE_URL': settings.SITE_URL}