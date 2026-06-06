from django.conf import settings
from django.db import models


class WatchlistItem(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="watchlist_items",
    )
    movie_id = models.IntegerField()
    movie_title = models.CharField(max_length=255)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-added_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "movie_id"], name="unique_user_watchlist_item"
            )
        ]
        indexes = [
            models.Index(fields=["user", "added_at"]),
            models.Index(fields=["movie_id"]),
        ]

    def __str__(self):
        return f"{self.user.username}: {self.movie_title}"


class RecommendationHistory(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recommendation_history",
    )
    query_title = models.CharField(max_length=255)
    selected_movie_id = models.IntegerField(null=True, blank=True)
    selected_movie_title = models.CharField(max_length=255, blank=True)
    recommendations_json = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} searched {self.query_title}"


class RecentlyViewed(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recently_viewed",
    )
    movie_id = models.IntegerField()
    movie_title = models.CharField(max_length=255)
    last_viewed_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-last_viewed_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "movie_id"], name="unique_user_recently_viewed_item"
            )
        ]
        indexes = [
            models.Index(fields=["user", "last_viewed_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} viewed {self.movie_title}"


class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    bio = models.TextField(blank=True, max_length=500)
    is_public = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["is_public"]),
        ]

    def __str__(self):
        return f"{self.user.username} profile"


class Follow(models.Model):
    follower = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="following",
    )
    following = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="followers",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["follower", "following"], name="unique_follow_pair"
            )
        ]
        indexes = [
            models.Index(fields=["follower", "created_at"]),
            models.Index(fields=["following", "created_at"]),
        ]

    def __str__(self):
        return f"{self.follower.username} follows {self.following.username}"


class Activity(models.Model):
    ACTION_CHOICES = [
        ("VIEWED", "Viewed a movie"),
        ("WATCHLIST_ADD", "Added to watchlist"),
        ("WATCHLIST_REMOVE", "Removed from watchlist"),
        ("FOLLOWED", "Followed a user"),
    ]
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="activities",
    )
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    target_movie_id = models.IntegerField(null=True, blank=True)
    target_movie_title = models.CharField(max_length=255, blank=True)
    target_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="targeted_activities",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} {self.action} {self.target_movie_title or ''}"
