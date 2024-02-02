from django.contrib import admin
from .models import FilesUpload,criminal_table,VideoUpload,CustomUser
from django.contrib.auth.admin import UserAdmin
# Register your models here.
admin.site.register(FilesUpload)
class Admincriminal_table(admin.ModelAdmin):
    list_display = ['name','criminal_id','image','case','Inperiod']
admin.site.register(criminal_table,Admincriminal_table)
admin.site.register(CustomUser)
admin.site.register(VideoUpload)