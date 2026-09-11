from django.contrib import admin

from .models import StudySession


@admin.register(StudySession)
class StudySessionAdmin(
    admin.ModelAdmin
):

    list_display = (
        "participant_code",
        "condition_order",
        "consent_given",
        "completed",
        "created_at"
    )

    list_filter = (
        "condition_order",
        "completed",
        "consent_given"
    )

    readonly_fields = (
        "participant_code",
        "created_at",
        "updated_at"
    )