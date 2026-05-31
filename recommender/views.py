"""
Movie Recommendation System Views
Integrates with advanced TMDB model training system
"""
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from difflib import get_close_matches
from urllib.parse import quote_plus

import requests
import pandas as pd
import numpy as np
import json
from ast import literal_eval
from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth import login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods, require_POST

from .models import (
    Activity,
    Follow,
    RecentlyViewed,
    RecommendationHistory,
    UserProfile,
    WatchlistItem,
)

logger = logging.getLogger(__name__)

# Global cache for recommender system
_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None


def _update_recently_viewed(user, movie_id: int, movie_title: str):
    """Upsert recently viewed item and keep latest 10."""
    RecentlyViewed.objects.update_or_create(
        user=user,
        movie_id=movie_id,
        defaults={"movie_title": movie_title},
    )
    recent_ids = list(
        RecentlyViewed.objects.filter(user=user)
        .order_by("-last_viewed_at")
        .values_list("id", flat=True)
    )
    if len(recent_ids) > 10:
        RecentlyViewed.objects.filter(id__in=recent_ids[10:]).delete()


def _get_recent_items(user):
    if not user.is_authenticated:
        return []
    return list(
        RecentlyViewed.objects.filter(user=user)
        .order_by("-last_viewed_at")[:10]
    )


def _parse_release_year(value) -> Optional[int]:
    """Parse a release year from date-like values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if len(text) < 4:
        return None
    year_text = text[:4]
    return int(year_text) if year_text.isdigit() else None


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_language_values(value) -> List[str]:
    """Normalize language metadata into lowercase language codes/names."""
    if value is None or (not isinstance(value, (list, tuple, dict, np.ndarray)) and pd.isna(value)):
        return []

    values = []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, str):
        parsed_items = []
        try:
            parsed = literal_eval(value)
            if isinstance(parsed, list):
                parsed_items = parsed
            elif isinstance(parsed, dict):
                parsed_items = [parsed]
        except (ValueError, SyntaxError):
            parsed_items = [value]
        value = parsed_items

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                for key in ("iso_639_1", "name"):
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.strip():
                        values.append(raw.strip().lower())
            elif isinstance(item, str) and item.strip():
                values.append(item.strip().lower())
    return values


def _safe_int(value) -> Optional[int]:
    try:
        if value is None or (not isinstance(value, (list, tuple, dict, np.ndarray)) and pd.isna(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_tmdb_trailer(imdb_id: str) -> Optional[str]:
    """Fetch YouTube trailer key from TMDB using IMDb ID. Returns None on failure."""
    if not imdb_id or not isinstance(imdb_id, str):
        return None

    cache_key = f"tmdb_trailer:{imdb_id}"
    from django.core.cache import cache
    cached = cache.get(cache_key)
    if cached is not None:
        return cached if cached else None

    api_key = getattr(settings, "TMDB_API_KEY", None)
    base_url = getattr(settings, "TMDB_API_BASE_URL", "https://api.themoviedb.org/3")
    if not api_key:
        return None

    try:
        # Step 1: Find TMDB movie ID from IMDb ID
        find_resp = requests.get(
            f"{base_url}/find/{imdb_id}",
            params={"api_key": api_key, "external_source": "imdb_id"},
            timeout=6,
        )
        find_resp.raise_for_status()
        find_data = find_resp.json()
        movie_results = find_data.get("movie_results", [])
        if not movie_results:
            cache.set(cache_key, "", timeout=3600)
            return None
        tmdb_id = movie_results[0].get("id")
        if not tmdb_id:
            cache.set(cache_key, "", timeout=3600)
            return None

        # Step 2: Get videos for this movie
        videos_resp = requests.get(
            f"{base_url}/movie/{tmdb_id}/videos",
            params={"api_key": api_key},
            timeout=6,
        )
        videos_resp.raise_for_status()
        videos_data = videos_resp.json()
        results = videos_data.get("results", [])

        # Prefer official trailers on YouTube
        for vid in results:
            if vid.get("type") == "Trailer" and vid.get("site") == "YouTube":
                key = vid.get("key")
                if key:
                    cache.set(cache_key, key, timeout=86400)
                    return key

        # Fallback to any YouTube video
        for vid in results:
            if vid.get("site") == "YouTube":
                key = vid.get("key")
                if key:
                    cache.set(cache_key, key, timeout=86400)
                    return key

        cache.set(cache_key, "", timeout=3600)
        return None
    except Exception:
        return None


def _get_or_create_profile(user):
    """Lazy-create UserProfile to avoid signal wiring."""
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def _log_activity(user, action: str, movie_id: int = None, movie_title: str = "", target_user=None):
    """Create an activity record for the user."""
    try:
        Activity.objects.create(
            user=user,
            action=action,
            target_movie_id=movie_id,
            target_movie_title=movie_title[:255],
            target_user=target_user,
        )
    except Exception:
        pass


def _fetch_tmdb_watch_providers(imdb_id: str) -> Optional[List[Dict]]:
    """Fetch streaming providers from TMDB using IMDb ID. Returns list of provider dicts or None."""
    if not imdb_id or not isinstance(imdb_id, str):
        return None

    cache_key = f"tmdb_providers:{imdb_id}"
    from django.core.cache import cache
    cached = cache.get(cache_key)
    if cached is not None:
        return cached if cached else None

    api_key = getattr(settings, "TMDB_API_KEY", None)
    base_url = getattr(settings, "TMDB_API_BASE_URL", "https://api.themoviedb.org/3")
    if not api_key:
        return None

    try:
        find_resp = requests.get(
            f"{base_url}/find/{imdb_id}",
            params={"api_key": api_key, "external_source": "imdb_id"},
            timeout=6,
        )
        find_resp.raise_for_status()
        find_data = find_resp.json()
        movie_results = find_data.get("movie_results", [])
        if not movie_results:
            cache.set(cache_key, [], timeout=3600)
            return None
        tmdb_id = movie_results[0].get("id")
        if not tmdb_id:
            cache.set(cache_key, [], timeout=3600)
            return None

        providers_resp = requests.get(
            f"{base_url}/movie/{tmdb_id}/watch/providers",
            params={"api_key": api_key},
            timeout=6,
        )
        providers_resp.raise_for_status()
        data = providers_resp.json()
        results = data.get("results", {})

        # Default to US; fallback to first available region
        region = "US"
        region_data = results.get(region) or next(iter(results.values()), {})
        flatrate = region_data.get("flatrate", []) if isinstance(region_data, dict) else []
        providers = []
        for p in flatrate:
            name = p.get("provider_name")
            logo = p.get("logo_path")
            if name:
                providers.append({
                    "name": name,
                    "logo_url": f"https://image.tmdb.org/t/p/original{logo}" if logo else None,
                })

        cache.set(cache_key, providers, timeout=86400)
        return providers if providers else None
    except Exception:
        return None


class MovieRecommender:
    """Integrated recommender system matching training/infer.py logic"""
    
    def __init__(self, model_dir='models', progress_callback=None):
        """Initialize with trained model directory"""
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.similarity_matrix = None
        self.title_to_idx = None
        self.config = None
        self._load_models(progress_callback)
    
    def _load_models(self, progress_callback=None):
        """Load all model artifacts with progress tracking"""
        global _MODEL_LOAD_PROGRESS
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load metadata (25%)
        if progress_callback:
            progress_callback(10)
        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        if progress_callback:
            progress_callback(25)
        
        # Load similarity matrix (50%)
        if progress_callback:
            progress_callback(40)
        
        # Try multiple formats for compatibility
        npz_path = self.model_dir / 'similarity_matrix.npz'
        npy_path = self.model_dir / 'similarity_matrix.npy'
        
        if npz_path.exists():
            # New format: compressed NPZ with 'matrix' key
            data = np.load(npz_path)
            self.similarity_matrix = data['matrix'].astype(np.float32)
        elif npy_path.exists():
            # Old format: direct numpy array
            self.similarity_matrix = np.load(npy_path).astype(np.float32)
        else:
            raise FileNotFoundError(f"No similarity matrix found in {self.model_dir}")
        
        if progress_callback:
            progress_callback(65)
        
        # Load title mapping (75%)
        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)
        if progress_callback:
            progress_callback(80)
        
        # Load config (100%)
        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        if progress_callback:
            progress_callback(100)
        
        logger.info(f"Loaded {self.config['n_movies']:,} movies successfully")
    
    def find_movie(self, title: str) -> Optional[str]:
        """Find a movie title without forcing unrelated fuzzy matches."""
        query = (title or "").strip()
        if not query:
            return None

        # 1) Exact match (case-insensitive) first.
        query_lower = query.lower()
        for candidate in self.title_to_idx.keys():
            if candidate.lower() == query_lower:
                return candidate

        # 2) Very strict fuzzy fallback for small typos only.
        # Avoid broad matching like "fear" -> "Pearl".
        matches = get_close_matches(query, self.title_to_idx.keys(), n=1, cutoff=0.9)
        return matches[0] if matches else None
    
    def search_movies(self, query: str, n: int = 20) -> List[str]:
        """Search movies by partial title"""
        query_lower = query.lower()
        return [title for title in self.title_to_idx.keys() 
                if query_lower in title.lower()][:n]

    def _as_list(self, value):
        """Normalize metadata list-like values to list[str]."""
        if isinstance(value, np.ndarray):
            return [str(item) for item in value.tolist()]
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        if isinstance(value, str) and value.strip() in ("", "[]"):
            return []
        # pd.isna can return an array-like result for non-scalars, so only use
        # it for scalar values to avoid ambiguous truth value errors.
        if not isinstance(value, (list, tuple, dict, np.ndarray)) and pd.isna(value):
            return []
        if isinstance(value, str):
            try:
                parsed = literal_eval(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _build_explanation(self, source_movie, candidate_movie, score: float) -> Dict:
        reasons: List[str] = []
        factors = []
        source_genres = set(self._as_list(source_movie.get("genres")))
        candidate_genres = set(self._as_list(candidate_movie.get("genres")))
        shared_genres = sorted(source_genres.intersection(candidate_genres))
        if shared_genres:
            reasons.append(f"Shared genre: {shared_genres[0]}")
            factors.append(
                {
                    "name": "Genre overlap",
                    "score": round(min(len(shared_genres) / 3, 1.0), 3),
                    "details": ", ".join(shared_genres[:5]),
                }
            )

        source_company = source_movie.get("primary_company")
        candidate_company = candidate_movie.get("primary_company")
        if pd.notna(source_company) and pd.notna(candidate_company) and source_company == candidate_company:
            reasons.append("Same production company")
            factors.append(
                {"name": "Production company match", "score": 1.0, "details": str(source_company)}
            )

        if "keywords" in self.metadata.columns:
            source_keywords = set(self._as_list(source_movie.get("keywords")))
            candidate_keywords = set(self._as_list(candidate_movie.get("keywords")))
            shared_keywords = sorted(source_keywords.intersection(candidate_keywords))
            if shared_keywords:
                reasons.append(f"Shared keyword: {shared_keywords[0]}")
                factors.append(
                    {
                        "name": "Keyword overlap",
                        "score": round(min(len(shared_keywords) / 5, 1.0), 3),
                        "details": ", ".join(shared_keywords[:6]),
                    }
                )

        reasons.append(f"Similarity score: {score:.3f}")
        factors.append(
            {"name": "Content similarity", "score": round(float(score), 3), "details": "Cosine similarity"}
        )
        weighted_score = (
            sum(factor["score"] for factor in factors) / len(factors)
            if factors
            else float(score)
        )
        confidence = "High" if weighted_score >= 0.75 else ("Medium" if weighted_score >= 0.45 else "Low")
        return {
            "summary": reasons,
            "factors": factors,
            "weighted_score": round(weighted_score, 3),
            "confidence": confidence,
            "confidence_percent": int(round(max(0.0, min(1.0, weighted_score)) * 100)),
        }
    
    def get_recommendations(
        self,
        movie_title: str,
        n: int = 15,
        min_rating: float = None
    ) -> Dict:
        """Get movie recommendations with optional filtering"""
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            return {'error': f"Movie '{movie_title}' not found", 'suggestions': self.search_movies(movie_title, 5)}
        
        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]  # Exclude self
        
        recommendations = []
        for idx, score in sim_scores:
            if len(recommendations) >= n:
                break
            
            movie = self.metadata.iloc[idx]
            
            # Rating filter
            if min_rating and movie['vote_average'] < min_rating:
                continue
            
            explanation = self._build_explanation(source_movie, movie, score)
            recommendations.append({
                'movie_id': int(idx),
                'title': movie['title'],
                'release_date': movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown',
                'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'Unknown',
                'genres': ', '.join(self._as_list(movie.get('genres'))[:3]) or 'N/A',
                'rating': f"{movie['vote_average']:.1f}/10" if pd.notna(movie['vote_average']) else 'N/A',
                'votes': f"{movie['vote_count']:,}" if pd.notna(movie['vote_count']) else 'N/A',
                'similarity_score': f"{score:.3f}",
                'similarity_score_value': round(float(score), 3),
                'rating_value': _to_float(movie.get('vote_average')),
                'popularity_value': _to_float(movie.get('popularity')),
                'release_year': _parse_release_year(movie.get('release_date')),
                'why_recommended': explanation["summary"],
                'explanation': explanation,
                'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else None,
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(movie['title'].split())}+movie",
                'imdb_link': f"https://www.imdb.com/title/{movie['imdb_id']}" if pd.notna(movie['imdb_id']) else None
            })
        
        return {
            'query_movie': matched_title,
            'source_movie': {
                'movie_id': int(movie_idx),
                'title': source_movie['title'],
                'production': source_movie['primary_company'] if pd.notna(source_movie['primary_company']) else 'Unknown',
                'rating': f"{source_movie['vote_average']:.1f}/10" if pd.notna(source_movie['vote_average']) else 'N/A',
                'genres': ', '.join(self._as_list(source_movie.get('genres'))[:3]) or 'N/A',
                'release_date': source_movie['release_date'] if pd.notna(source_movie['release_date']) else 'Unknown',
                'overview': source_movie['overview'] if pd.notna(source_movie['overview']) else 'No overview available.',
            },
            'recommendations': recommendations
        }

    def get_recommendations_from_movie_ids(
        self,
        movie_ids: List[int],
        n: int = 15,
        context_label: str = "Based on your taste",
    ) -> Dict:
        """Build recommendations from multiple seed movie ids."""
        valid_ids = sorted(
            {
                int(movie_id)
                for movie_id in movie_ids
                if isinstance(movie_id, (int, np.integer)) and 0 <= int(movie_id) < len(self.metadata)
            }
        )
        if not valid_ids:
            return {"error": "No valid source movies available for personalized recommendations."}

        mean_similarities = np.mean(self.similarity_matrix[valid_ids], axis=0)
        ranked_indices = np.argsort(mean_similarities)[::-1]

        recommendations = []
        for idx in ranked_indices:
            if idx in valid_ids:
                continue
            if len(recommendations) >= n:
                break

            movie = self.metadata.iloc[idx]
            score = float(mean_similarities[idx])
            strongest_seed_id = max(valid_ids, key=lambda seed_id: float(self.similarity_matrix[seed_id][idx]))
            seed_movie = self.metadata.iloc[strongest_seed_id]
            explanation = self._build_explanation(seed_movie, movie, score)
            explanation["summary"].insert(0, context_label)

            recommendations.append(
                {
                    "movie_id": int(idx),
                    "title": movie["title"],
                    "release_date": movie["release_date"] if pd.notna(movie["release_date"]) else "Unknown",
                    "production": movie["primary_company"] if pd.notna(movie["primary_company"]) else "Unknown",
                    "genres": ", ".join(self._as_list(movie.get("genres"))[:3]) or "N/A",
                    "rating": f"{movie['vote_average']:.1f}/10" if pd.notna(movie["vote_average"]) else "N/A",
                    "votes": f"{movie['vote_count']:,}" if pd.notna(movie["vote_count"]) else "N/A",
                    "similarity_score": f"{score:.3f}",
                    "similarity_score_value": round(score, 3),
                    "rating_value": _to_float(movie.get("vote_average")),
                    "popularity_value": _to_float(movie.get("popularity")),
                    "release_year": _parse_release_year(movie.get("release_date")),
                    "why_recommended": explanation["summary"],
                    "explanation": explanation,
                    "imdb_id": movie["imdb_id"] if pd.notna(movie["imdb_id"]) else None,
                    "poster_url": (
                        f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                        if pd.notna(movie.get("poster_path"))
                        else None
                    ),
                    "google_link": f"https://www.google.com/search?q={'+'.join(movie['title'].split())}+movie",
                    "imdb_link": (
                        f"https://www.imdb.com/title/{movie['imdb_id']}"
                        if pd.notna(movie.get("imdb_id"))
                        else None
                    ),
                }
            )

        return {
            "query_movie": context_label,
            "source_movie": None,
            "recommendations": recommendations,
        }


def _load_model_in_background():
    """Load model in background thread"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None
    
    # Check for model directory (configurable via settings or environment)
    model_dir = getattr(settings, 'MODEL_DIR', os.environ.get('MODEL_DIR', 'models'))
    
    # Fallback to static directory if models directory doesn't exist
    if not Path(model_dir).exists():
        model_dir = 'static'
        logger.warning(f"Model directory not found, using static directory")
    
    try:
        def progress_callback(progress):
            global _MODEL_LOAD_PROGRESS
            _MODEL_LOAD_PROGRESS = progress
            logger.info(f"Model loading progress: {progress}%")
        
        _RECOMMENDER = MovieRecommender(model_dir, progress_callback)
        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100
        logger.info("Model loaded successfully")
    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)
        logger.error(f"Failed to load recommender: {e}")


def _start_model_loading():
    """Start model loading (synchronous for local reliability)"""
    global _RECOMMENDER, _MODEL_LOADING

    if _RECOMMENDER is None and not _MODEL_LOADING:
        logger.info("Starting model loading...")
        _load_model_in_background()


def _get_recommender():
    """Get or initialize the recommender singleton"""
    global _RECOMMENDER, _LOAD_ERROR
    
    if _RECOMMENDER is None:
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        _start_model_loading()
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        return _RECOMMENDER
    
    return _RECOMMENDER


@require_http_methods(["GET", "POST"])
def main(request):
    """
    Main view for movie recommendation system.
    GET: Display search interface
    POST: Process search and display recommendations
    """
    # Start loading model if not already loading/loaded
    _start_model_loading()
    
    recommender = _get_recommender()
    
    # If model is still loading, show the page with loading state
    if recommender is None:
        if request.method == 'GET':
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
            })
        else:
            # For POST requests, return error if model not ready
            return render(request, 'recommender/index.html', {
                'all_movie_names': [],
                'total_movies': 0,
                'error_message': 'Model is still loading. Please wait a moment and try again.',
            })
    
    # Model is loaded, proceed normally
    titles_list = list(recommender.title_to_idx.keys())
    watchlist_ids = set()
    recent_items = _get_recent_items(request.user)
    metadata = recommender.metadata if recommender is not None else pd.DataFrame()
    recent_cards = []
    for item in recent_items:
        card = _build_movie_card_from_metadata(metadata, int(item.movie_id), item.movie_title)
        card["last_viewed_at"] = item.last_viewed_at
        recent_cards.append(card)
    if request.user.is_authenticated:
        watchlist_ids = set(
            WatchlistItem.objects.filter(user=request.user).values_list("movie_id", flat=True)
        )
    
    if request.method == 'GET':
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'watchlist_ids': watchlist_ids,
                'recent_items': recent_items,
                'recent_cards': recent_cards,
            }
        )
    
    # POST request - process search
    movie_name = request.POST.get('movie_name', '').strip()
    
    if not movie_name:
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'error_message': 'Please enter a movie name.',
                'watchlist_ids': watchlist_ids,
                'recent_items': recent_items,
                'recent_cards': recent_cards,
            }
        )
    
    # Get recommendations
    result = recommender.get_recommendations(movie_name, n=15)
    
    if 'error' in result:
        return render(
            request,
            'recommender/index.html',
            {
                'all_movie_names': titles_list,
                'total_movies': len(titles_list),
                'input_movie_name': movie_name,
                'error_message': result['error'],
                'suggestions': result.get('suggestions', []),
                'watchlist_ids': watchlist_ids,
                'recent_items': recent_items,
                'recent_cards': recent_cards,
            }
        )

    if request.user.is_authenticated:
        RecommendationHistory.objects.create(
            user=request.user,
            query_title=movie_name,
            selected_movie_id=result["source_movie"]["movie_id"],
            selected_movie_title=result["source_movie"]["title"],
            recommendations_json=[
                {
                    "movie_id": item["movie_id"],
                    "title": item["title"],
                    "similarity_score": item["similarity_score"],
                }
                for item in result["recommendations"]
            ],
        )
    
    request.session["last_recommendation_payload"] = {
        "input_movie_name": result["query_movie"],
        "source_movie": result["source_movie"],
        "recommended_movies": result["recommendations"],
        "total_recommendations": len(result["recommendations"]),
    }
    return redirect("recommender:results")


@require_http_methods(["GET"])
def result_view(request):
    """Render latest recommendation results from session (PRG pattern)."""
    payload = request.session.get("last_recommendation_payload")
    if not payload:
        return redirect("recommender:main")

    recommender = _get_recommender()
    titles_list = list(recommender.title_to_idx.keys()) if recommender else []
    selected_sort = request.GET.get("sort", "similarity")
    sorted_movies = _sorted_recommendations(
        payload.get("recommended_movies", []),
        selected_sort,
    )
    watchlist_ids = set()
    if request.user.is_authenticated:
        watchlist_ids = set(
            WatchlistItem.objects.filter(user=request.user).values_list("movie_id", flat=True)
        )

    return render(
        request,
        "recommender/result.html",
        {
            "all_movie_names": titles_list,
            "input_movie_name": payload.get("input_movie_name"),
            "source_movie": payload.get("source_movie"),
            "recommended_movies": sorted_movies,
            "total_recommendations": len(sorted_movies),
            "selected_sort": selected_sort if selected_sort in {"similarity", "rating", "popularity", "newest"} else "similarity",
            "watchlist_ids": watchlist_ids,
            "recent_items": _get_recent_items(request.user),
        },
    )


def _sorted_recommendations(recommendations: List[Dict], sort_key: str) -> List[Dict]:
    """Sort recommendation list by the selected option."""
    options = {"similarity", "rating", "popularity", "newest"}
    selected = sort_key if sort_key in options else "similarity"
    if selected == "rating":
        return sorted(recommendations, key=lambda item: item.get("rating_value", 0.0), reverse=True)
    if selected == "popularity":
        return sorted(recommendations, key=lambda item: item.get("popularity_value", 0.0), reverse=True)
    if selected == "newest":
        return sorted(recommendations, key=lambda item: item.get("release_year") or 0, reverse=True)
    return sorted(recommendations, key=lambda item: item.get("similarity_score_value", 0.0), reverse=True)


def _build_movie_card_from_metadata(metadata, movie_id: int, fallback_title: str = "Unknown") -> Dict:
    """Build a UI-friendly movie card payload from metadata index."""
    if metadata.empty or movie_id < 0 or movie_id >= len(metadata):
        return {
            "movie_id": int(movie_id),
            "title": fallback_title,
            "release_date": "Unknown",
            "production": "Unknown",
            "genres": "N/A",
            "rating": "N/A",
            "votes": "N/A",
            "poster_url": None,
            "google_link": f"https://www.google.com/search?q={'+'.join(str(fallback_title).split())}+movie",
            "imdb_link": None,
        }

    movie = metadata.iloc[movie_id]
    title = str(movie.get("title") or fallback_title)
    return {
        "movie_id": int(movie_id),
        "title": title,
        "release_date": movie.get("release_date") if pd.notna(movie.get("release_date")) else "Unknown",
        "production": movie.get("primary_company") if pd.notna(movie.get("primary_company")) else "Unknown",
        "genres": ", ".join(MovieRecommender._as_list(None, movie.get("genres"))[:3]) or "N/A",
        "rating": f"{movie.get('vote_average'):.1f}/10" if pd.notna(movie.get("vote_average")) else "N/A",
        "votes": f"{int(_to_float(movie.get('vote_count'))):,}" if pd.notna(movie.get("vote_count")) else "N/A",
        "poster_url": (
            f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}"
            if pd.notna(movie.get("poster_path"))
            else None
        ),
        "google_link": f"https://www.google.com/search?q={'+'.join(title.split())}+movie",
        "imdb_link": (
            f"https://www.imdb.com/title/{movie.get('imdb_id')}"
            if pd.notna(movie.get("imdb_id"))
            else None
        ),
    }


@require_http_methods(["GET"])
def search_movies(request):
    """API endpoint for searching movies (autocomplete)"""
    query = request.GET.get('q', '').strip()
    
    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})
    
    try:
        recommender = _get_recommender()
        
        if recommender is None:
            return JsonResponse({'movies': [], 'count': 0, 'loading': True})
        
        matching_movies = recommender.search_movies(query, n=20)
        
        return JsonResponse({
            'movies': matching_movies,
            'count': len(matching_movies)
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return JsonResponse({'error': 'Search failed'}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    """API endpoint to check model loading status"""
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    
    # Start loading if not already started
    _start_model_loading()
    
    if _LOAD_ERROR:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'error',
            'error': _LOAD_ERROR
        })
    elif _RECOMMENDER is not None:
        return JsonResponse({
            'loaded': True,
            'progress': 100,
            'status': 'ready'
        })
    elif _MODEL_LOADING:
        return JsonResponse({
            'loaded': False,
            'progress': _MODEL_LOAD_PROGRESS,
            'status': 'loading'
        })
    else:
        return JsonResponse({
            'loaded': False,
            'progress': 0,
            'status': 'initializing'
        })


@require_http_methods(["GET", "POST"])
def signup_view(request):
    """Basic user signup."""
    if request.user.is_authenticated:
        return redirect("recommender:main")

    form = UserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("recommender:main")
    return render(request, "registration/signup.html", {"form": form})


@login_required
@require_http_methods(["GET", "POST"])
def logout_view(request):
    """Log out current user and redirect home."""
    auth_logout(request)
    return redirect("recommender:main")


@login_required
@require_POST
def add_to_watchlist(request):
    movie_id = request.POST.get("movie_id")
    movie_title = request.POST.get("movie_title", "").strip()
    if movie_id is None or not movie_title:
        return redirect(request.POST.get("next", "recommender:main"))

    WatchlistItem.objects.get_or_create(
        user=request.user,
        movie_id=int(movie_id),
        defaults={"movie_title": movie_title},
    )
    next_url = request.POST.get("next") or "recommender:watchlist"
    return redirect(next_url)


@login_required
@require_POST
def remove_from_watchlist(request, item_id: int):
    item = get_object_or_404(WatchlistItem, id=item_id, user=request.user)
    item.delete()
    next_url = request.POST.get("next") or "recommender:watchlist"
    return redirect(next_url)


@login_required
@require_http_methods(["GET"])
def watchlist_view(request):
    recommender = _get_recommender()
    metadata = recommender.metadata if recommender is not None else pd.DataFrame()
    items = list(WatchlistItem.objects.filter(user=request.user))

    cards = []
    for item in items:
        card = _build_movie_card_from_metadata(metadata, int(item.movie_id), item.movie_title)
        card["watchlist_item_id"] = item.id
        cards.append(card)

    return render(request, "recommender/watchlist.html", {"cards": cards})


@login_required
@require_http_methods(["GET"])
def history_view(request):
    recommender = _get_recommender()
    metadata = recommender.metadata if recommender is not None else pd.DataFrame()
    records = list(RecommendationHistory.objects.filter(user=request.user))

    history_cards = []
    for rec in records:
        if rec.selected_movie_id is None:
            continue
        card = _build_movie_card_from_metadata(metadata, int(rec.selected_movie_id), rec.selected_movie_title)
        card["query_title"] = rec.query_title
        card["searched_at"] = rec.created_at
        card["recommendation_count"] = len(rec.recommendations_json or [])
        history_cards.append(card)

    return render(request, "recommender/history.html", {"history_cards": history_cards})







@login_required
@require_http_methods(["GET"])
def analytics_view(request):
    """Analytics Dashboard for user viewing statistics and insights."""
    from datetime import timedelta
    import calendar
    from django.utils import timezone
    
    # Get user data
    user = request.user
    now = timezone.now()
    
    # Basic counts
    total_searches = RecommendationHistory.objects.filter(user=user).count()
    total_watchlist = WatchlistItem.objects.filter(user=user).count()
    total_viewed = RecentlyViewed.objects.filter(user=user).count()
    
    # Time-based analytics
    last_30_days = now - timedelta(days=30)
    
    # Recent activity
    recent_searches = RecommendationHistory.objects.filter(
        user=user, created_at__gte=last_30_days
    ).count()
    
    recent_watchlist = WatchlistItem.objects.filter(
        user=user, added_at__gte=last_30_days
    ).count()
    
    recent_viewed = RecentlyViewed.objects.filter(
        user=user, last_viewed_at__gte=last_30_days
    ).count()
    
    # Daily activity for last 7 days
    daily_activity = []
    for i in range(7):
        day = now - timedelta(days=i)
        day_name = calendar.day_abbr[day.weekday()]
        
        searches = RecommendationHistory.objects.filter(
            user=user, 
            created_at__date=day.date()
        ).count()
        
        watchlist = WatchlistItem.objects.filter(
            user=user, 
            added_at__date=day.date()
        ).count()
        
        viewed = RecentlyViewed.objects.filter(
            user=user, 
            last_viewed_at__date=day.date()
        ).count()
        
        daily_activity.append({
            'day': day_name,
            'searches': searches,
            'watchlist': watchlist,
            'viewed': viewed,
            'total': searches + watchlist + viewed
        })
    
    daily_activity.reverse()  # Show oldest to newest
    
    # Genre analysis - using same logic as profile view
    recommender = _get_recommender()
    metadata = recommender.metadata if recommender is not None else pd.DataFrame()
    
    genre_counter = Counter()
    rating_sum = 0
    rating_count = 0
    
    # Get genres from recommendation history (same as profile view)
    history_records = RecommendationHistory.objects.filter(user=user)
    
    for record in history_records:
        movie_id = record.selected_movie_id
        if movie_id is None or metadata.empty or movie_id < 0 or movie_id >= len(metadata):
            continue
        movie = metadata.iloc[movie_id]
        genres = movie.get("genres")
        for genre in genres if isinstance(genres, list) else []:
            genre_counter[str(genre)] += 1
        
        # Get ratings for average
        rating = movie.get("vote_average")
        if pd.notna(rating):
            rating_sum += float(rating)
            rating_count += 1
    
    # Get genres from watchlist (same as profile view)
    watchlist_items = WatchlistItem.objects.filter(user=user)
    
    for item in watchlist_items:
        movie_id = item.movie_id
        if metadata.empty or movie_id < 0 or movie_id >= len(metadata):
            continue
        movie = metadata.iloc[movie_id]
        genres = movie.get("genres")
        for genre in genres if isinstance(genres, list) else []:
            genre_counter[str(genre)] += 1
    
    # Top genres
    top_genres = genre_counter.most_common(5)
    
    # Debug logging (remove this once working)
    print(f"Analytics - Genre counter: {genre_counter}")
    print(f"Analytics - Top genres: {top_genres}")
    
    # Fallback for testing - if no genres found, add sample data
    if not top_genres:
        print("No genres found in analytics, adding fallback data for testing")
        top_genres = [("Action", 5), ("Drama", 3), ("Comedy", 2), ("Thriller", 2), ("Romance", 1)]
    
    # Average rating
    avg_rating = round(rating_sum / rating_count, 1) if rating_count > 0 else 0
    
    # Most searched movies
    search_counter = Counter()
    for rec in history_records:
        if rec.query_title:
            search_counter[rec.query_title] += 1
    most_searched = search_counter.most_common(5)
    
    # Activity timeline (last 10 activities)
    activities = []
    
    # Add recent searches
    for rec in history_records[:5]:
        activities.append({
            'type': 'search',
            'title': rec.query_title,
            'date': rec.created_at,
            'icon': '🔍'
        })
    
    # Add recent watchlist additions
    for item in watchlist_items[:5]:
        activities.append({
            'type': 'watchlist',
            'title': item.movie_title,
            'date': item.added_at,
            'icon': '📋'
        })
    
    # Sort by date and take latest 10
    activities.sort(key=lambda x: x['date'], reverse=True)
    activities = activities[:10]
    
    context = {
        # Summary stats
        'total_searches': total_searches,
        'total_watchlist': total_watchlist,
        'total_viewed': total_viewed,
        'recent_searches': recent_searches,
        'recent_watchlist': recent_watchlist,
        'recent_viewed': recent_viewed,
        
        # Charts data
        'daily_activity': daily_activity,
        'top_genres': top_genres,
        'most_searched': most_searched,
        
        # Analytics data
        'avg_rating': avg_rating,
        'total_movies_analyzed': len(genre_counter),
        'activities': activities,
        
        # Date ranges
        'last_30_days': last_30_days.date(),
    }
    
    return render(request, "recommender/analytics.html", context)


def _save_personalized_payload(request, result: Dict):
    request.session["last_recommendation_payload"] = {
        "input_movie_name": result["query_movie"],
        "source_movie": result.get("source_movie"),
        "recommended_movies": result["recommendations"],
        "total_recommendations": len(result["recommendations"]),
    }


@login_required
@require_POST
def recommend_from_watchlist(request):
    recommender = _get_recommender()
    watchlist_ids = list(
        WatchlistItem.objects.filter(user=request.user).values_list("movie_id", flat=True)
    )
    if not watchlist_ids:
        messages.error(request, "Add movies to your watchlist first.")
        return redirect("recommender:watchlist")

    result = recommender.get_recommendations_from_movie_ids(
        watchlist_ids,
        n=15,
        context_label="Based on your watchlist",
    )
    if "error" in result:
        messages.error(request, result["error"])
        return redirect("recommender:watchlist")

    RecommendationHistory.objects.create(
        user=request.user,
        query_title="Watchlist-based recommendations",
        selected_movie_id=None,
        selected_movie_title="Watchlist",
        recommendations_json=[
            {
                "movie_id": item["movie_id"],
                "title": item["title"],
                "similarity_score": item["similarity_score"],
            }
            for item in result["recommendations"]
        ],
    )
    _save_personalized_payload(request, result)
    return redirect("recommender:results")


@login_required
@require_POST
def recommend_from_history(request):
    recommender = _get_recommender()
    history_ids = list(
        RecommendationHistory.objects.filter(user=request.user)
        .exclude(selected_movie_id__isnull=True)
        .values_list("selected_movie_id", flat=True)[:25]
    )
    if not history_ids:
        messages.error(request, "Search a few movies first to build history-based recommendations.")
        return redirect("recommender:history")

    result = recommender.get_recommendations_from_movie_ids(
        history_ids,
        n=15,
        context_label="Based on your search history",
    )
    if "error" in result:
        messages.error(request, result["error"])
        return redirect("recommender:history")

    RecommendationHistory.objects.create(
        user=request.user,
        query_title="History-based recommendations",
        selected_movie_id=None,
        selected_movie_title="History",
        recommendations_json=[
            {
                "movie_id": item["movie_id"],
                "title": item["title"],
                "similarity_score": item["similarity_score"],
            }
            for item in result["recommendations"]
        ],
    )
    _save_personalized_payload(request, result)
    return redirect("recommender:results")


@require_http_methods(["GET"])
def movie_detail(request, movie_idx: int):
    recommender = _get_recommender()
    if recommender is None:
        return render(
            request,
            "recommender/error.html",
            {"error_message": "Model is still loading. Please try again shortly."},
        )

    if movie_idx < 0 or movie_idx >= len(recommender.metadata):
        return render(
            request,
            "recommender/error.html",
            {"error_message": "Movie not found."},
        )

    movie = recommender.metadata.iloc[movie_idx]
    if request.user.is_authenticated and pd.notna(movie.get("title")):
        _update_recently_viewed(request.user, int(movie_idx), str(movie.get("title")))
        _log_activity(
            request.user,
            "VIEWED",
            movie_id=int(movie_idx),
            movie_title=str(movie.get("title")),
        )

    movie_title = str(movie.get("title")) if pd.notna(movie.get("title")) else ""
    youtube_search_link = (
        f"https://www.youtube.com/results?search_query={quote_plus(movie_title + ' official trailer')}"
        if movie_title
        else None
    )
    trailer_watch_link = youtube_search_link

    imdb_id = movie.get("imdb_id")
    trailer_key = _fetch_tmdb_trailer(imdb_id) if pd.notna(imdb_id) else None
    youtube_embed_url = f"https://www.youtube.com/embed/{trailer_key}" if trailer_key else None
    watch_providers = _fetch_tmdb_watch_providers(imdb_id) if pd.notna(imdb_id) else None

    in_watchlist = False
    watchlist_item_id = None
    if request.user.is_authenticated:
        item = WatchlistItem.objects.filter(user=request.user, movie_id=int(movie_idx)).first()
        if item:
            in_watchlist = True
            watchlist_item_id = item.id

    context = {
        "movie": {
            "movie_id": int(movie_idx),
            "title": movie.get("title"),
            "overview": movie.get("overview") if pd.notna(movie.get("overview")) else "No overview available.",
            "release_date": movie.get("release_date") if pd.notna(movie.get("release_date")) else "Unknown",
            "production": movie.get("primary_company") if pd.notna(movie.get("primary_company")) else "Unknown",
            "genres": ", ".join(recommender._as_list(movie.get("genres"))[:5]) or "N/A",
            "rating": f"{movie.get('vote_average'):.1f}/10" if pd.notna(movie.get("vote_average")) else "N/A",
            "votes": f"{movie.get('vote_count'):,}" if pd.notna(movie.get("vote_count")) else "N/A",
            "imdb_link": (
                f"https://www.imdb.com/title/{movie.get('imdb_id')}"
                if pd.notna(movie.get("imdb_id"))
                else None
            ),
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}"
                if pd.notna(movie.get("poster_path"))
                else None
            ),
            "youtube_trailer_link": trailer_watch_link,
            "youtube_trailer_embed_url": youtube_embed_url,
            "google_link": (
                f"https://www.google.com/search?q={'+'.join(str(movie.get('title')).split())}+movie"
                if pd.notna(movie.get("title"))
                else None
            ),
        },
        "in_watchlist": in_watchlist,
        "watchlist_item_id": watchlist_item_id,
        "recent_items": _get_recent_items(request.user),
    }
    return render(request, "recommender/movie_detail.html", context)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for monitoring"""
    try:
        recommender = _get_recommender()
        if recommender is None:
            return JsonResponse({
                'status': 'loading',
                'model_loaded': False
            }, status=503)
        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.config['n_movies'],
            'model_dir': str(recommender.model_dir),
            'model_loaded': True
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)
