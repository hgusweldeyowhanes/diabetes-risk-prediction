from django.conf import settings
from django.db import models


class Patient(models.Model):
    """Patient record for longitudinal tracking of predictions."""

    SEX_CHOICES = [
        ("F", "Female"),
        ("M", "Male"),
        ("U", "Unknown"),
    ]

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="patients",
    )
    external_id = models.CharField(max_length=64, blank=True, db_index=True)
    first_name = models.CharField(max_length=100, blank=True)
    last_name = models.CharField(max_length=100, blank=True)
    sex = models.CharField(max_length=1, choices=SEX_CHOICES, default="U")
    date_of_birth = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        name = f"{self.first_name} {self.last_name}".strip() or "Unnamed"
        return f"{name} ({self.external_id or self.pk})"
