import uuid

from django.db import models


class StudySession(models.Model):

    ORDER_CHOICES = [
        ("pairwise_first", "Pairwise first"),
        ("ranking_first", "Ranking first"),
    ]

    participant_code = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False
    )

    condition_order = models.CharField(
        max_length=32,
        choices=ORDER_CHOICES
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    consent_given = models.BooleanField(
        default=False
    )

    current_stage = models.CharField(
        max_length=32,
        default="intro"
    )

    # Movies selected for this participant
    pairwise_pairs = models.JSONField(
        default=list,
        blank=True
    )

    ranking_sets = models.JSONField(
        default=list,
        blank=True
    )

    evaluation_pairs = models.JSONField(
        default=list,
        blank=True
    )

    # User responses
    pairwise_responses = models.JSONField(
        default=list,
        blank=True
    )

    ranking_responses = models.JSONField(
        default=list,
        blank=True
    )

    evaluation_responses = models.JSONField(
        default=list,
        blank=True
    )

    questionnaire_responses = models.JSONField(
        default=dict,
        blank=True
    )

    # Estimated preference vectors
    pairwise_weights = models.JSONField(
        default=list,
        blank=True
    )

    ranking_weights = models.JSONField(
        default=list,
        blank=True
    )

    # Evaluation results
    pairwise_metrics = models.JSONField(
        default=dict,
        blank=True
    )

    ranking_metrics = models.JSONField(
        default=dict,
        blank=True
    )

    # Completion time
    pairwise_time_seconds = models.FloatField(
        default=0.0
    )

    ranking_time_seconds = models.FloatField(
        default=0.0
    )

    completed = models.BooleanField(
        default=False
    )

    def __str__(self):
        return f"Project 4 session {self.participant_code}"