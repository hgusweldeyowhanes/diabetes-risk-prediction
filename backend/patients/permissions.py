from rest_framework.permissions import BasePermission


class IsPatientOwner(BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.created_by_id == getattr(request.user, "id", None)
