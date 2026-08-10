from django.conf import settings
from django.db import models


class Profile(models.Model):
    """Extended user profile for clinicians / researchers."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile"
    )
    organization = models.CharField(max_length=200, blank=True)
    role = models.CharField(
        max_length=40,
        choices=[
            ("viewer", "Viewer"),
            ("clinician", "Clinician"),
            ("admin", "Admin"),
        ],
        default="viewer",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.role})"
