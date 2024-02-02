from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

# Create your models here.
class FilesUpload(models.Model):
    file=models.FileField()


class criminal_table(models.Model):
    name = models.CharField(max_length=1000, blank=False)
    criminal_id = models.CharField(max_length=200,blank=False)
    image = models.ImageField(upload_to='img/%Y/%m/%d', blank=False)
    case = models.CharField(max_length=10000, blank=False)
    Inperiod = models.CharField(max_length=10, blank=False)
    area=models.CharField(max_length=100,default='1000sqft')
    
class Video(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')

    def __str__(self):
        return self.title
    

class VideoUpload(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')

    def __str__(self):
        return self.title

class CustomUserManager(BaseUserManager):
    def create_user(self, full_name, phone, username, password=None, **extra_fields):
        if not username:
            raise ValueError('The Username field must be set')
        user = self.model(
            full_name=full_name,
            phone=phone,
            username=username,
            **extra_fields
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        return self.create_user(username=username, password=password, **extra_fields)

class CustomUser(AbstractBaseUser, PermissionsMixin):
    full_name = models.CharField(max_length=255)
    phone = models.CharField(max_length=15)
    username = models.CharField(max_length=30, unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = CustomUserManager()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['full_name', 'phone']

    def __str__(self):
        return self.username

