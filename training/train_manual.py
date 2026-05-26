"""
Manual implementation of a movie recommender training pipeline using TF-IDF and SVD without scikit-learn.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval
import pickle
import json
import warnings
from collections import Counter, defaultdict
import re
import math

warnings.filterwarnings('ignore')


class ManualTfidfVectorizer:
    """Manual TF-IDF implementation without scikit-learn."""
    
    def __init__(self, max_features=20000, ngram_range=(1, 2), min_df=3, max_df=0.7, stop_words='english', sublinear_tf=True):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = self._load_stop_words() if stop_words == 'english' else set()
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}
        self.idf_ = None
        self.n_docs = 0
        
    def _load_stop_words(self):
        """Common English stop words."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'our', 'their', 'what', 'which', 'who', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'now', 'then', 'once', 'here', 'there'
        }
    
    def _tokenize(self, text):
        """Simple tokenizer - lowercase, extract words."""
        return re.findall(r'\b[a-z]+\b', text.lower())
    
    def _get_ngrams(self, tokens, n):
        """Generate n-grams from tokens."""
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _preprocess(self, text):
        """Tokenize and extract n-grams."""
        tokens = self._tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        
        features = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            features.extend(self._get_ngrams(tokens, n))
        return features
    
    def fit_transform(self, documents):
        """Build vocabulary and transform documents to TF-IDF matrix."""
        self.n_docs = len(documents)
        
        # First pass: collect document frequency for all terms
        doc_freq = defaultdict(int)
        doc_features = []
        
        for doc in documents:
            features = self._preprocess(doc)
            doc_features.append(features)
            unique_features = set(features)
            for f in unique_features:
                doc_freq[f] += 1
        
        # Filter by min_df and max_df, select top features
        min_df_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * self.n_docs)
        max_df_count = int(self.max_df * self.n_docs) if isinstance(self.max_df, float) else self.max_df
        
        filtered_terms = [
            (term, freq) for term, freq in doc_freq.items()
            if min_df_count <= freq <= max_df_count
        ]
        
        # Sort by document frequency and select top features
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = filtered_terms[:self.max_features]
        
        # Build vocabulary mapping
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(top_terms)}
        
        # Compute IDF
        self.idf_ = np.zeros(len(self.vocabulary_))
        for term, idx in self.vocabulary_.items():
            df = doc_freq[term]
            # Smooth IDF: log((1 + n_docs) / (1 + df)) + 1
            self.idf_[idx] = math.log((1 + self.n_docs) / (1 + df)) + 1
        
        # Second pass: build TF matrix
        return self._build_matrix(doc_features)
    
    def _build_matrix(self, doc_features):
        """Build sparse TF-IDF matrix from document features."""
        data = []
        row_idx = []
        col_idx = []
        
        for doc_id, features in enumerate(doc_features):
            # Count term frequencies
            term_counts = Counter(features)
            
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    col = self.vocabulary_[term]
                    # Sublinear TF scaling: 1 + log(tf)
                    if self.sublinear_tf:
                        tf = 1 + math.log(count) if count > 0 else 0
                    else:
                        tf = count
                    
                    # TF-IDF weighting
                    weight = tf * self.idf_[col]
                    
                    data.append(weight)
                    row_idx.append(doc_id)
                    col_idx.append(col)
        
        # Create COO format sparse matrix
        return SparseMatrix(data, (self.n_docs, len(self.vocabulary_)), row_idx, col_idx)
    
    def transform(self, documents):
        """Transform new documents using fitted vocabulary."""
        doc_features = [self._preprocess(doc) for doc in documents]
        return self._build_matrix(doc_features)


class SparseMatrix:
    """Simple COO format sparse matrix implementation."""
    
    def __init__(self, data, shape, row, col):
        self.data = np.array(data, dtype=np.float32)
        self.row = np.array(row, dtype=np.int32)
        self.col = np.array(col, dtype=np.int32)
        self.shape = shape
        
    def toarray(self):
        """Convert to dense numpy array."""
        arr = np.zeros(self.shape, dtype=np.float32)
        for r, c, d in zip(self.row, self.col, self.data):
            arr[r, c] = d
        return arr
    
    def dot(self, other):
        """Matrix multiplication with another sparse or dense matrix."""
        if isinstance(other, SparseMatrix):
            # COO @ COO -> dense
            return self.toarray() @ other.toarray()
        else:
            return self.toarray() @ other
    
    def multiply(self, scalar):
        """Element-wise multiplication by scalar."""
        self.data *= scalar
        return self
    
    def transpose(self):
        """Return transposed matrix."""
        return SparseMatrix(self.data, (self.shape[1], self.shape[0]), self.col, self.row)
    
    @property
    def nnz(self):
        """Number of non-zero elements."""
        return len(self.data)


class ManualTruncatedSVD:
    """Manual Truncated SVD using randomized algorithm."""
    
    def __init__(self, n_components=500, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.singular_values_ = None
        
    def fit_transform(self, X):
        """Fit SVD and return transformed matrix."""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        n_components = min(self.n_components, n_samples, n_features)
        
        # Randomized SVD: Halko et al. 2009
        # 1. Generate random matrix
        Omega = np.random.randn(n_features, n_components).astype(np.float32)
        
        # 2. Sample column space: Y = X @ Omega
        Y = X @ Omega
        
        # 3. Power iterations for accuracy
        for _ in range(2):
            Y = X @ (X.T @ Y)
        
        # 4. QR decomposition to get orthonormal basis
        Q, _ = np.linalg.qr(Y)
        
        # 5. Project X onto Q: B = Q.T @ X
        B = Q.T @ X
        
        # 6. SVD of small matrix B
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)
        
        # 7. Reconstruct U
        U = Q @ U_hat
        
        # Keep top components
        self.components_ = Vt[:n_components]
        self.singular_values_ = S[:n_components]
        self.explained_variance_ratio_ = (S[:n_components] ** 2) / np.sum(S ** 2)
        
        # Return reduced representation: U * S
        return U[:, :n_components] * S[:n_components]


def cosine_similarity_manual(X, Y=None):
    """Compute cosine similarity between rows of X and Y."""
    if Y is None:
        Y = X
    
    # L2 normalize
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    
    # Cosine similarity = dot product of normalized vectors
    return X_norm @ Y_norm.T


class SnowballStemmer:
    """Simple Porter-like stemmer for English."""
    
    def __init__(self, language='english'):
        self.language = language
        
    def stem(self, word):
        """Apply simple stemming rules."""
        word = word.lower()
        
        # Simple suffix removal rules (subset of Porter stemmer)
        suffixes = [
            ('ies', 'y'), ('ied', 'y'), ('ying', 'y'),
            ('ied', 'y'), ('ies', 'y'),
            ('s', ''), ('es', ''), ('ed', ''), ('ing', ''),
            ('ly', ''), ('ment', ''), ('ness', ''),
            ('tion', 't'), ('sion', 's'),
            ('able', ''), ('ible', ''), ('ful', ''),
            ('er', ''), ('or', ''), ('ist', ''),
            ('ism', ''), ('ize', ''), ('ise', ''),
        ]
        
        for suffix, replacement in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)] + replacement
        
        return word


class ManualMovieRecommenderTrainer:
    """Training pipeline with manual algorithm implementations."""
    
    def __init__(self, output_dir='./models', use_dimensionality_reduction=True, n_components=500):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_svd = use_dimensionality_reduction
        self.n_components = n_components
        self.stemmer = SnowballStemmer('english')
        self.tfidf_vectorizer = None
        self.svd_model = None
        
    def load_data(self, data_path):
        """Load TMDB dataset from CSV file."""
        print("Loading TMDB dataset...")
        
        if Path(data_path).is_file():
            df = pd.read_csv(data_path, low_memory=False)
        else:
            csv_path = Path(data_path) / 'TMDB_movie_dataset_v11.csv'
            df = pd.read_csv(csv_path, low_memory=False)
        
        print(f"Loaded {len(df)} movies")
        return df
    
    def parse_json_column(self, col_data, key='name'):
        """Parse JSON-like string columns."""
        if pd.isna(col_data) or col_data == '' or col_data == '[]':
            return []
        
        try:
            parsed = literal_eval(col_data) if isinstance(col_data, str) else col_data
            if isinstance(parsed, list):
                return [item[key] for item in parsed if isinstance(item, dict) and key in item]
            return []
        except:
            if isinstance(col_data, str):
                return [item.strip() for item in col_data.split(',') if item.strip()]
            return []
    
    def clean_and_engineer_features(self, df, quality_threshold='medium'):
        """Feature engineering pipeline."""
        print("Engineering features...")
        
        thresholds = {'low': 5, 'medium': 50, 'high': 500}
        min_votes = thresholds.get(quality_threshold, 50)
        df = df[df['vote_count'] >= min_votes].copy()
        
        df = df[df['status'] == 'Released'].copy()
        
        print("Parsing genres, keywords, and production companies...")
        df['genres'] = df['genres'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['keywords'] = df['keywords'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['companies'] = df['production_companies'].apply(lambda x: self.parse_json_column(x, 'name'))
        df['countries'] = df['production_countries'].apply(lambda x: self.parse_json_column(x, 'name'))
        
        df['primary_company'] = df['companies'].apply(lambda x: x[0] if x else None)
        
        df['overview_clean'] = df['overview'].fillna('').astype(str)
        df['overview_words'] = df['overview_clean'].apply(
            lambda x: [word.lower() for word in x.split()[:50]]
        )
        
        df['tagline_clean'] = df['tagline'].fillna('').astype(str)
        df['tagline_words'] = df['tagline_clean'].apply(
            lambda x: [word.lower() for word in x.split()]
        )
        
        df['keywords'] = df['keywords'].apply(
            lambda x: [self.stemmer.stem(kw.lower().replace(" ", "")) for kw in x[:15]]
        )
        
        df['genres'] = df['genres'].apply(
            lambda x: [genre.lower().replace(" ", "") for genre in x]
        )
        
        df['companies_weighted'] = df['companies'].apply(
            lambda x: [x[0].lower().replace(" ", "")] * 2 if x and len(x) > 0 else []
        )
        df['companies_clean'] = df['companies'].apply(
            lambda x: [comp.lower().replace(" ", "") for comp in x[:3]]
        )
        
        df['countries_clean'] = df['countries'].apply(
            lambda x: [country.lower().replace(" ", "") for country in x[:2]]
        )
        
        df['soup'] = (
            df['keywords'] + 
            df['genres'] * 2 +
            df['companies_weighted'] + 
            df['companies_clean'] +
            df['countries_clean'] +
            df['overview_words'] +
            df['tagline_words']
        )
        df['soup'] = df['soup'].apply(lambda x: ' '.join(x) if x else '')
        
        df = df[df['soup'].str.len() > 20].copy()
        df = df.dropna(subset=['title'])
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        df['quality_score'] = df['vote_average'] * np.log1p(df['vote_count'])
        df = df.sort_values('quality_score', ascending=False)
        
        if 'tconst' in df.columns and 'imdb_id' not in df.columns:
            df['imdb_id'] = df['tconst']
        
        df = df.reset_index(drop=True)
        
        print(f"Processed {len(df)} valid movies")
        return df
    
    def build_tfidf_matrix(self, df):
        """Build TF-IDF matrix with manual implementation."""
        print("Building TF-IDF matrix...")
        
        n_movies = len(df)
        if n_movies < 10000:
            max_features = 10000
        elif n_movies < 100000:
            max_features = 15000
        else:
            max_features = 20000
        
        print(f"Using max_features={max_features} for {n_movies} movies")
        
        self.tfidf_vectorizer = ManualTfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.7,
            stop_words='english',
            sublinear_tf=True
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['soup'].tolist())
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
        print(f"Matrix sparsity: {sparsity:.2f}%")
        
        return tfidf_matrix
    
    def compute_similarity_matrix(self, tfidf_matrix):
        """Compute similarity with optional SVD dimensionality reduction."""
        if self.use_svd and tfidf_matrix.shape[0] > 1000:
            print(f"Applying SVD dimensionality reduction to {self.n_components} components...")
            
            n_components = min(
                self.n_components,
                tfidf_matrix.shape[0] - 1,
                tfidf_matrix.shape[1] - 1
            )
            
            self.svd_model = ManualTruncatedSVD(n_components=n_components, random_state=42)
            
            # Convert to dense for SVD (memory intensive but required for randomized SVD)
            dense_matrix = tfidf_matrix.toarray()
            reduced_matrix = self.svd_model.fit_transform(dense_matrix)
            
            print(f"Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.3f}")
            print(f"Reduced matrix shape: {reduced_matrix.shape}")
            
            # Compute cosine similarity on reduced matrix
            print("Computing cosine similarity matrix...")
            similarity_matrix = cosine_similarity_manual(reduced_matrix)
            
            return similarity_matrix, reduced_matrix
        else:
            print("Computing cosine similarity matrix on full TF-IDF...")
            dense_matrix = tfidf_matrix.toarray()
            similarity_matrix = cosine_similarity_manual(dense_matrix)
            return similarity_matrix, dense_matrix
    
    def train(self, data_path, quality_threshold='medium', max_movies=None):
        """Full training pipeline."""
        # Load and preprocess data
        df = self.load_data(data_path)
        df = self.clean_and_engineer_features(df, quality_threshold)
        
        if max_movies and len(df) > max_movies:
            df = df.head(max_movies)
            print(f"Limited to top {max_movies} movies by quality score")
        
        # Build TF-IDF matrix
        tfidf_matrix = self.build_tfidf_matrix(df)
        
        # Compute similarity matrix
        similarity_matrix, reduced_matrix = self.compute_similarity_matrix(tfidf_matrix)
        
        # Save models
        print("Saving models...")
        self._save_models(df, similarity_matrix, reduced_matrix)
        
        print(f"Training complete! Processed {len(df)} movies.")
        return df, similarity_matrix
    
    def _save_models(self, df, similarity_matrix, reduced_matrix):
        """Save all model artifacts."""
        # Save metadata
        metadata_cols = ['title', 'vote_average', 'vote_count', 'release_date', 
                        'genres', 'production_companies', 'imdb_id', 'overview',
                        'popularity', 'runtime', 'tagline', 'keywords', 'companies',
                        'countries_clean', 'poster_path']
        
        metadata = df[[c for c in metadata_cols if c in df.columns]].copy()
        metadata.to_parquet(self.output_dir / 'movie_metadata.parquet')
        
        # Save similarity matrix
        np.savez_compressed(self.output_dir / 'similarity_matrix.npz', 
                          matrix=similarity_matrix.astype(np.float16))
        
        # Save title to index mapping
        title_to_idx = {title: idx for idx, title in enumerate(df['title'].values)}
        with open(self.output_dir / 'title_to_idx.json', 'w') as f:
            json.dump(title_to_idx, f)
        
        # Save TF-IDF vectorizer
        with open(self.output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save SVD model
        
        if self.svd_model:
            with open(self.output_dir / 'svd_model.pkl', 'wb') as f:
                pickle.dump(self.svd_model, f)
        
        # Save config

        config = {
            'n_movies': len(df),
            'use_svd': self.use_svd,
            'n_components': self.n_components if self.svd_model else None,
            'explained_variance': self.svd_model.explained_variance_ratio_.sum() if self.svd_model else None,
            'vocab_size': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else None,
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Models saved to {self.output_dir}")


if __name__ == '__main__':
    # Example usage
    trainer = ManualMovieRecommenderTrainer(
        output_dir='./models',
        use_dimensionality_reduction=True,
        n_components=500
    )
    
    df, sim_matrix = trainer.train(
        'path/to/your/dataset.csv',
        quality_threshold='medium',
        max_movies=100000
    )
    
    print(f"\nFinal similarity matrix shape: {sim_matrix.shape}")
    print(f"Memory usage: {sim_matrix.nbytes / 1024 / 1024:.2f} MB")
