import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie_recommendation.settings')
django.setup()

from django.contrib.auth.models import User

# Create or get superuser
username = 'admin'
email = 'rojemmaharjan@gmail.com'
password = 'admin123'  
user, created = User.objects.get_or_create(
    username=username,
    defaults={
        'email': email,
        'is_staff': True,
        'is_superuser': True,
    }
)

if created:
    user.set_password(password)
    user.save()
    print(f"Superuser '{username}' created successfully!")
else:
    user.set_password(password)
    user.save()
    print(f"Password updated for '{username}'!")

print(f"Login with: {username} / {password}")
print(f"Access admin panel at: http://127.0.0.1:8000/admin/")
