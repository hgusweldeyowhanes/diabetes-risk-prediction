from django.contrib import admin

from .models import Patient


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("id", "first_name", "last_name", "external_id", "created_by", "updated_at")
    search_fields = ("first_name", "last_name", "external_id")
