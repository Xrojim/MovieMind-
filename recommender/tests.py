from unittest.mock import patch

import pandas as pd
import numpy as np
from django.contrib.auth.models import User
from django.test import TestCase
from django.test import override_settings
from django.urls import reverse

from .models import RecommendationHistory, RecentlyViewed, WatchlistItem


class FakeRecommender:
    def __init__(self):
        self.title_to_idx = {"The Matrix": 0, "Inception": 1}
        self.metadata = pd.DataFrame(
            [
                {
                    "title": "The Matrix",
                    "release_date": "1999-01-01",
                    "primary_company": "Warner",
                    "genres": ["Action", "Sci-Fi"],
                    "vote_average": 8.7,
                    "vote_count": 1000,
                    "popularity": 80.0,
                    "original_language": "en",
                    "overview": "A hacker discovers reality.",
                    "imdb_id": "tt0133093",
                    "poster_path": "/matrix.jpg",
                },
                {
                    "title": "Inception",
                    "release_date": "2010-01-01",
                    "primary_company": "Warner",
                    "genres": ["Action", "Sci-Fi"],
                    "vote_average": 8.5,
                    "vote_count": 900,
                    "popularity": 95.0,
                    "original_language": "en",
                    "overview": "Dreams in dreams.",
                    "imdb_id": "tt1375666",
                    "poster_path": "/inception.jpg",
                },
            ]
        )
        self.similarity_matrix = np.array([[1.0, 0.912], [0.912, 1.0]])

    def search_movies(self, query, n=20):
        return [title for title in self.title_to_idx if query.lower() in title.lower()][:n]

    def get_recommendations(self, movie_name, n=15):
        if movie_name != "The Matrix":
            return {"error": "not found"}
        return {
            "query_movie": "The Matrix",
            "source_movie": {
                "movie_id": 0,
                "title": "The Matrix",
                "production": "Warner",
                "rating": "8.7/10",
                "genres": "Action, Sci-Fi",
                "release_date": "1999-01-01",
                "overview": "A hacker discovers reality.",
            },
            "recommendations": [
                {
                    "movie_id": 1,
                    "title": "Inception",
                    "release_date": "2010-01-01",
                    "production": "Warner",
                    "genres": "Action, Sci-Fi",
                    "rating": "8.5/10",
                    "votes": "900",
                    "similarity_score": "0.912",
                    "similarity_score_value": 0.912,
                    "rating_value": 8.5,
                    "popularity_value": 95.0,
                    "release_year": 2010,
                    "why_recommended": ["Shared genre: Action", "Same production company", "Similarity score: 0.912"],
                    "explanation": {
                        "summary": ["Shared genre: Action", "Same production company", "Similarity score: 0.912"],
                        "confidence": "High",
                        "confidence_percent": 85,
                        "factors": [
                            {"name": "Genre overlap", "score": 0.5, "details": "Action, Sci-Fi"},
                            {"name": "Content similarity", "score": 0.912, "details": "Cosine similarity"},
                        ],
                    },
                    "google_link": "https://google.com",
                    "imdb_link": "https://imdb.com",
                }
            ],
        }

    def get_recommendations_from_movie_ids(self, movie_ids, n=15, context_label="Based on your taste"):
        if not movie_ids:
            return {"error": "No source movies"}
        return {
            "query_movie": context_label,
            "source_movie": None,
            "recommendations": [
                {
                    "movie_id": 1,
                    "title": "Inception",
                    "release_date": "2010-01-01",
                    "production": "Warner",
                    "genres": "Action, Sci-Fi",
                    "rating": "8.5/10",
                    "votes": "900",
                    "similarity_score": "0.912",
                    "similarity_score_value": 0.912,
                    "rating_value": 8.5,
                    "popularity_value": 95.0,
                    "release_year": 2010,
                    "why_recommended": [context_label, "Similarity score: 0.912"],
                    "explanation": {
                        "summary": [context_label, "Similarity score: 0.912"],
                        "confidence": "High",
                        "confidence_percent": 85,
                        "factors": [{"name": "Content similarity", "score": 0.912, "details": "Cosine similarity"}],
                    },
                    "google_link": "https://google.com",
                    "imdb_link": "https://imdb.com",
                }
            ],
        }


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)
class FeatureFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="demo", password="pass12345")

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_movie_detail_page(self, _start, mock_get):
        mock_get.return_value = FakeRecommender()
        response = self.client.get(reverse("recommender:movie_detail", args=[0]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The Matrix")
        self.assertContains(response, "YouTube Trailer")
        self.assertContains(response, "youtube.com/embed")

    def test_signup_page_loads(self):
        response = self.client.get(reverse("recommender:signup"))
        self.assertEqual(response.status_code, 200)

    def test_signup_creates_account_and_logs_in(self):
        response = self.client.post(
            reverse("recommender:signup"),
            {
                "username": "newuser",
                "password1": "StrongPass123!",
                "password2": "StrongPass123!",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("recommender:main"))
        self.assertTrue(User.objects.filter(username="newuser").exists())

    def test_watchlist_requires_login(self):
        response = self.client.get(reverse("recommender:watchlist"))
        self.assertEqual(response.status_code, 302)

    def test_watchlist_add_remove(self):
        self.client.login(username="demo", password="pass12345")
        add_resp = self.client.post(
            reverse("recommender:add_to_watchlist"),
            {"movie_id": 1, "movie_title": "Inception", "next": reverse("recommender:watchlist")},
        )
        self.assertEqual(add_resp.status_code, 302)
        self.assertEqual(WatchlistItem.objects.filter(user=self.user).count(), 1)

        with patch("recommender.views._get_recommender", return_value=FakeRecommender()), patch(
            "recommender.views._start_model_loading"
        ):
            watchlist_page = self.client.get(reverse("recommender:watchlist"))
            self.assertEqual(watchlist_page.status_code, 200)
            self.assertContains(watchlist_page, "Poster unavailable")
            self.assertContains(watchlist_page, "Details")

        item = WatchlistItem.objects.get(user=self.user)
        remove_resp = self.client.post(
            reverse("recommender:remove_from_watchlist", args=[item.id]),
            {"next": reverse("recommender:watchlist")},
        )
        self.assertEqual(remove_resp.status_code, 302)
        self.assertEqual(WatchlistItem.objects.filter(user=self.user).count(), 0)

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_history_saved_on_recommendation(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        mock_get.return_value = FakeRecommender()
        response = self.client.post(reverse("recommender:main"), {"movie_name": "The Matrix"})
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("recommender:results"))
        self.assertEqual(RecommendationHistory.objects.filter(user=self.user).count(), 1)

        results_response = self.client.get(reverse("recommender:results"))
        self.assertEqual(results_response.status_code, 200)
        self.assertContains(results_response, "Shared genre: Action")
        self.assertContains(results_response, "Show details")
        self.assertContains(results_response, "Sort by")
        self.assertContains(results_response, "Confidence:")

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_results_supports_sort_query_param(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        mock_get.return_value = FakeRecommender()
        self.client.post(reverse("recommender:main"), {"movie_name": "The Matrix"})
        response = self.client.get(reverse("recommender:results"), {"sort": "rating"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'option value="rating" selected')

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_recently_viewed_upsert_and_cap(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        fake = FakeRecommender()
        rows = []
        for i in range(12):
            rows.append(
                {
                    "title": f"Movie {i}",
                    "release_date": "2020-01-01",
                    "primary_company": "Studio",
                    "genres": ["Drama"],
                    "vote_average": 7.0,
                    "vote_count": 100,
                    "overview": "Overview",
                    "imdb_id": f"tt0000{i}",
                }
            )
        fake.metadata = pd.DataFrame(rows)
        mock_get.return_value = fake

        for i in range(12):
            self.client.get(reverse("recommender:movie_detail", args=[i]))

        self.assertEqual(RecentlyViewed.objects.filter(user=self.user).count(), 10)
        latest = RecentlyViewed.objects.filter(user=self.user).first()
        self.assertEqual(latest.movie_title, "Movie 11")

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_continue_browsing_context_on_index(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        mock_get.return_value = FakeRecommender()
        RecentlyViewed.objects.create(user=self.user, movie_id=0, movie_title="The Matrix")
        response = self.client.get(reverse("recommender:main"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Continue Browsing")

    
    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_history_card_grid_shows_movie_details(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        mock_get.return_value = FakeRecommender()
        RecommendationHistory.objects.create(
            user=self.user,
            query_title="The Matrix",
            selected_movie_id=0,
            selected_movie_title="The Matrix",
            recommendations_json=[{"movie_id": 1, "title": "Inception"}],
        )
        response = self.client.get(reverse("recommender:history"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Query: The Matrix")
        self.assertContains(response, "1 recs")

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_recommend_from_watchlist_redirects_to_results(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        mock_get.return_value = FakeRecommender()
        WatchlistItem.objects.create(user=self.user, movie_id=0, movie_title="The Matrix")

        response = self.client.post(reverse("recommender:recommend_from_watchlist"))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("recommender:results"))

        results_response = self.client.get(reverse("recommender:results"))
        self.assertEqual(results_response.status_code, 200)
        self.assertContains(results_response, "Based on your watchlist")

    @patch("recommender.views._get_recommender")
    @patch("recommender.views._start_model_loading")
    def test_recommend_from_history_redirects_to_results(self, _start, mock_get):
        self.client.login(username="demo", password="pass12345")
        mock_get.return_value = FakeRecommender()
        RecommendationHistory.objects.create(
            user=self.user,
            query_title="The Matrix",
            selected_movie_id=0,
            selected_movie_title="The Matrix",
            recommendations_json=[],
        )

        response = self.client.post(reverse("recommender:recommend_from_history"))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("recommender:results"))

        results_response = self.client.get(reverse("recommender:results"))
        self.assertEqual(results_response.status_code, 200)
        self.assertContains(results_response, "Based on your search history")
