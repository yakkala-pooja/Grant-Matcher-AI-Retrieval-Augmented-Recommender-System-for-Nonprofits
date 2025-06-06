import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
import os
from data_loader import load_grants_data, prepare_data_for_embedding
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter, defaultdict
import json
from functools import lru_cache
import logging

logging.basicConfig( # Configure logging
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GrantRecommender:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Grant Recommender system.
        Args: model_name (str): Name of the SentenceTransformer model to use
        """
        logger.info(f"Initializing GrantRecommender with model: {model_name}")
        self.model_name = model_name
        self.model = None
        self.index = None
        self.grants_data = None
        self.grant_ids = None
        self.embeddings = None
        self.tfidf = TfidfVectorizer(
            stop_words='english', ngram_range=(1, 2),
            max_features=1000)
        self.grant_tfidf_matrix = None
        self.funder_preferences = None
        self._term_pattern = re.compile(
            r'\b\w+(?:tion|ing|ment|ship)\b|\b\w+ (?:program|service|support)s?\b')
        logger.info("GrantRecommender initialized")
        
    def _init_model(self):
        """Initialize the SentenceTransformer model if not already initialized."""
        if self.model is None:
            logger.info("Initializing SentenceTransformer model...")
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing SentenceTransformer model: {str(e)}", exc_info=True)
                raise RuntimeError(f"Error initializing SentenceTransformer model: {str(e)}")
    
    def _load_funder_preferences(self, funder_history_dir: str = 'data') -> None:
        """Load and process funder preferences from history files."""
        if self.funder_preferences is not None:
            return 
        self.funder_preferences = defaultdict(lambda: {'ntee_codes': Counter(), 'keywords': Counter()})
        try:
            with open(os.path.join(funder_history_dir, 'funder_history.json'), 'r') as f:
                for funder_info in json.load(f):
                    funder_file = os.path.join(funder_history_dir, funder_info['filename'])
                    if not os.path.exists(funder_file):
                        continue
                    with open(funder_file, 'r') as f:
                        funder_data = json.load(f)
                        self._process_funder_data(funder_data)
        except Exception as e:
            logger.warning(f"Error loading funder preferences: {e}")
            self.funder_preferences = None

    def _process_funder_data(self, funder_data: Dict) -> None:
        funder = funder_data['funder_name']
        for org in funder_data['funded_organizations']:
            if ntee_code := org.get('ntee_code', '').strip():
                self.funder_preferences[funder]['ntee_codes'][ntee_code] += 1
            if purpose := org.get('purpose', '').lower():
                self.funder_preferences[funder]['keywords'].update(self._extract_key_terms(purpose))
    
    @lru_cache(maxsize=1024)
    def _calculate_boost(self, funder: str, ntee_code: str, mission: str) -> float:
        """Calculate ranking boost based on funder preferences with caching."""
        if not self.funder_preferences or funder not in self.funder_preferences:
            return 0.0
        prefs = self.funder_preferences[funder]
        ntee_boost = 0.15 * prefs['ntee_codes'][ntee_code] / sum(prefs['ntee_codes'].values()) if ntee_code else 0
        keywords = self._extract_key_terms(mission)
        keyword_boost = min(0.15, 0.1 * sum(
            prefs['keywords'][kw] / sum(prefs['keywords'].values()) for kw in keywords))
        return min(ntee_boost + keyword_boost, 0.3)
    
    @lru_cache(maxsize=1024)
    def _extract_key_terms(self, text: str) -> frozenset:
        """Extract key terms from text using precompiled regex patterns with caching."""
        return frozenset(self._term_pattern.findall(text.lower())) if text else frozenset()
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        logger.info("Computing embeddings...")
        self._init_model()  # Ensure model is initialized
        try: # Use larger batch size for efficiency
            embeddings = self.model.encode(
                texts, show_progress_bar=True, batch_size=64,  # Increased from 32
                normalize_embeddings=True, convert_to_tensor=False  # Ensure numpy output
            )
            logger.info(f"Successfully computed embeddings with shape {embeddings.shape}")
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}", exc_info=True)
            raise
        
    @lru_cache(maxsize=512)
    def _generate_justification(self, mission: str, grant_title: str, grant_description: str, grant_funding_type: str, award_ceiling: float, score: float) -> Dict:
        """
        Generate detailed justification for why a grant matches a mission statement.
        Uses caching for repeated matches and optimized text processing.
        """
        grant_text = f"{grant_title} {grant_description}"
        common_terms = self._extract_key_terms(mission) & self._extract_key_terms(grant_text)
        vectors = self.tfidf.transform([mission, grant_text])
        feature_names = self.tfidf.get_feature_names_out()
        shared_keywords = {
            feature_names[i] for i, v in enumerate(vectors[0].toarray()[0])
            if v > 0.1} & {
            feature_names[i] for i, v in enumerate(vectors[1].toarray()[0])
            if v > 0.1
        }
        return {
            'similarity_score': score, 'common_focus_areas': list(common_terms),
            'shared_keywords': list(shared_keywords),
            'alignment_summary': [
                f"Both focus on: {', '.join(common_terms)}" if common_terms else None,
                f"Shared themes: {', '.join(shared_keywords)}" if shared_keywords else None],
            'funding_alignment': [
                f"Grant offers funding up to ${award_ceiling:,.2f}"
                if award_ceiling and not pd.isna(award_ceiling) else None,
                f"Funding type: {grant_funding_type}"
                if grant_funding_type else None]
        }
        
    def fit(self, data_path: str) -> None:
        """Build the FAISS index from grant descriptions."""
        self.grants_data = load_grants_data(data_path) # Load and preprocess data
        texts, self.grant_ids = prepare_data_for_embedding(self.grants_data)
        self.embeddings = self._compute_embeddings(texts)
        self.tfidf.fit(texts) # Fit TF-IDF vectorizer with larger n_jobs
        self.grant_tfidf_matrix = self.tfidf.transform(texts)
        dimension = self.embeddings.shape[1] # Initialize and train FAISS index efficiently
        if len(texts) < 10000:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.train(self.embeddings)
        self.index.add(self.embeddings)
        
    def get_recommendations(self, mission_statement: str, nonprofit_info: Dict, 
                            top_n: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Get top N matching grants for a given mission statement."""
        if self.index is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        self._load_funder_preferences() # Initial setup
        mission_embedding = self._compute_embeddings([mission_statement])
        search_k = min(top_n * 20, len(self.grant_ids)) #  Configure search parameters
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = 10
        # Perform search
        distances, indices = self.index.search(mission_embedding, search_k)
        similarities = 1 / (1 + distances[0])
        valid_results = [(idx, sim) for idx, sim in zip(indices[0], similarities) if idx != -1]
        results = [] # Process results
        for idx, sim_score in valid_results:
            if len(results) >= top_n:
                break
            grant = self.grants_data[ # Get grant info
                self.grants_data['opportunity_number'] == self.grant_ids[idx]
            ].iloc[0].to_dict()
            boost = self._calculate_boost( # Calculate boost and check thresholds
                grant['agency_name'].lower(), nonprofit_info.get('ntee_code', ''),
                nonprofit_info['mission_statement'])
            score = sim_score * (1 + boost)
            if score < min_similarity: # Apply filters
                continue
            justification = self._generate_justification( # Generate match justification
                mission=mission_statement, grant_title=grant['opportunity_title'],
                grant_description=grant['grant_description'], grant_funding_type=grant.get('funding_instrument_type', ''),
                award_ceiling=grant.get('award_ceiling'), score=score)
            if boost > 0: # Add boost explanation if applicable
                justification['alignment_summary'].append(
                    f"Historical funding patterns suggest good fit (+{boost*100:.1f}% boost)")
            results.append({ # Build result
                'opportunity_number': self.grant_ids[idx], 'title': grant['opportunity_title'],
                'description': grant['grant_description'], 'agency': grant['agency_name'],
                'award_ceiling': grant.get('award_ceiling'), 'award_floor': grant.get('award_floor'),
                'funding_instrument_type': grant.get('funding_instrument_type'), 'category': grant.get('category'),
                'post_date': grant.get('post_date'), 'close_date': grant.get('close_date'),
                'similarity_score': float(score), 'original_similarity': float(sim_score),
                'funder_boost': float(boost), 'match_justification': justification})
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    def save_model(self, directory: str) -> None:
        """Save the model and index to disk efficiently."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin")) # Save FAISS index
        with open(os.path.join(directory, "model_data.pkl"), 'wb') as f: # Save other data efficiently
            pickle.dump({
                'grants_data': self.grants_data,
                'grant_ids': self.grant_ids, 'embeddings': self.embeddings,
                'tfidf': self.tfidf, 'grant_tfidf_matrix': self.grant_tfidf_matrix
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    @classmethod
    def load_model(cls, directory: str) -> 'GrantRecommender':
        """Load a saved model from disk efficiently."""
        logger.info(f"Loading model from directory: {directory}")
        recommender = cls()
        try:
            index_path = os.path.join(directory, "faiss_index.bin") # Load FAISS index
            logger.info(f"Loading FAISS index from {index_path}")
            recommender.index = faiss.read_index(index_path)
            logger.info("FAISS index loaded successfully")
            data_path = os.path.join(directory, "model_data.pkl") # Load other data efficiently
            logger.info(f"Loading model data from {data_path}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                recommender.grants_data = data['grants_data']
                recommender.grant_ids = data['grant_ids']
                recommender.embeddings = data['embeddings']
                recommender.tfidf = data['tfidf']
                recommender.grant_tfidf_matrix = data['grant_tfidf_matrix']
            logger.info("Model data loaded successfully")
            return recommender
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise 