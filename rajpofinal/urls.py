# webcam_project/urls.py
from django.contrib import admin
from django.urls import path
from rajpofinalapp.views import video_feed,vid_inp,signin,signup,home,history,analysis,update_anomaly
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video_feed/', video_feed, name='video_feed'),
    path('vid_inp/', vid_inp, name='vid_inp'),
    path('', signin, name="signin"),
    path('signup', signup, name="signup"),
    path('home', home, name="home"),
    path('admin/rajpofinalapp/videoupload/',history, name="history"),
    path('analysis',analysis, name="analysis"),
    path('update_anomaly/', update_anomaly, name='update_anomaly'),
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
