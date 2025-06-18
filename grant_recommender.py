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
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the recommender with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data = None
        self.embeddings = None
        self.grant_ids = None
        self.tfidf = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.grant_tfidf_matrix = None
        logger.info(f"GrantRecommender initialized with model: {model_name}")

    def fit(self, data_path: str) -> None:
        """Build the FAISS index from grant descriptions."""
        try:
            # Load and preprocess data
            self.grants_data = load_grants_data(data_path)
            texts, self.grant_ids = prepare_data_for_embedding(self.grants_data)
            
            # Generate embeddings
            self.embeddings = self._compute_embeddings(texts)
            
            # Fit TF-IDF vectorizer
            self.tfidf.fit(texts)
            self.grant_tfidf_matrix = self.tfidf.transform(texts)
            
            # Initialize and train FAISS index
            dimension = self.embeddings.shape[1]
            if len(texts) < 10000:
                self.index = faiss.IndexFlatL2(dimension)
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            
            logger.info(f"Successfully loaded and fit model with {len(texts)} grants")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Successfully computed embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            raise

    def save_model(self, output_dir):
        """Save the model data and FAISS index."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(output_dir, 'faiss_index.bin')
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save model data
            data_path = os.path.join(output_dir, 'model_data.pkl')
            model_data = {
                'grants_data': self.grants_data,
                'grant_ids': self.grant_ids,
                'grant_tfidf_matrix': self.grant_tfidf_matrix,
                'tfidf_vocabulary': self.tfidf.vocabulary_,
                'tfidf_idf': self.tfidf.idf_
            }
            with open(data_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved model data to {data_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, model_dir):
        """Load a pre-trained model from directory."""
        try:
            recommender = cls()
            
            # Load FAISS index
            index_path = os.path.join(model_dir, 'faiss_index.bin')
            recommender.index = faiss.read_index(index_path)
            logger.info("FAISS index loaded successfully")
            
            # Load model data
            data_path = os.path.join(model_dir, 'model_data.pkl')
            with open(data_path, 'rb') as f:
                model_data = pickle.load(f)
            
            recommender.grants_data = model_data['grants_data']
            recommender.grant_ids = model_data['grant_ids']
            recommender.grant_tfidf_matrix = model_data['grant_tfidf_matrix']
            recommender.tfidf.vocabulary_ = model_data['tfidf_vocabulary']
            recommender.tfidf.idf_ = model_data['tfidf_idf']
            
            logger.info("Model data loaded successfully")
            return recommender
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def find_matches(self, query, k=5):
        """Find the top k matching grants for a query."""
        try:
            if not self.index:
                raise ValueError("Model not loaded. Call load_model first.")
            
            # Encode query
            query_vector = self.model.encode([query])[0]
            query_vector = query_vector.reshape(1, -1)
            
            # Get more candidates for reranking
            k_candidates = min(k * 3, len(self.grant_ids))
            distances, indices = self.index.search(query_vector, k_candidates)
            
            # Convert L2 distances to cosine similarities
            # Using modified formula for better score distribution
            semantic_similarities = np.exp(-distances[0] / 4)  # Exponential scaling
            
            # Get TF-IDF score for query
            query_tfidf = self.tfidf.transform([query])
            
            # Compute TF-IDF similarities for candidates
            matches = []
            
            # Track max TF-IDF score for normalization
            max_tfidf_score = float('-inf')
            candidate_matches = []
            
            for idx, semantic_score in zip(indices[0], semantic_similarities):
                if idx < 0 or idx >= len(self.grant_ids):
                    continue
                
                # Get TF-IDF similarity
                grant_tfidf = self.grant_tfidf_matrix[idx]
                tfidf_score = float((query_tfidf * grant_tfidf.T).toarray()[0][0])
                max_tfidf_score = max(max_tfidf_score, tfidf_score)
                
                # Get grant data
                grant_id = self.grant_ids[idx]
                grant_data = self.grants_data[self.grants_data['opportunity_number'] == grant_id].iloc[0]
                
                # Clean up title and description
                title = grant_data.get('opportunity_title', '').strip()
                desc = grant_data.get('grant_description', '').strip()
                
                # If title is too long or looks like a description, try to extract a shorter title
                if len(title) > 100 or title.lower().startswith('notice of'):
                    # Try to extract a cleaner title from the first sentence
                    sentences = title.split('.')
                    title = sentences[0].strip()
                    if len(sentences) > 1:
                        desc = ' '.join(sentences[1:]).strip() + ' ' + desc
                
                match = grant_data.to_dict()
                match['opportunity_title'] = title
                match['grant_description'] = desc
                match['semantic_score'] = semantic_score
                match['tfidf_score'] = tfidf_score
                candidate_matches.append(match)
            
            # Normalize TF-IDF scores and compute final scores
            if max_tfidf_score > 0:
                for match in candidate_matches:
                    # Normalize TF-IDF score
                    match['tfidf_score'] = match['tfidf_score'] / max_tfidf_score
                    
                    # Apply sigmoid scaling to both scores for better distribution
                    semantic_scaled = 1 / (1 + np.exp(-6 * (match['semantic_score'] - 0.5)))
                    tfidf_scaled = 1 / (1 + np.exp(-6 * (match['tfidf_score'] - 0.5)))
                    
                    # Combine scores (70% semantic + 30% TF-IDF)
                    match['similarity_score'] = (0.7 * semantic_scaled) + (0.3 * tfidf_scaled)
                    
                    # Update individual scores for display
                    match['semantic_score'] = semantic_scaled
                    match['tfidf_score'] = tfidf_scaled
                    
                    # Generate match justification
                    match['match_justification'] = self._generate_justification(
                        query,
                        match['opportunity_title'],
                        match['grant_description'],
                        match.get('funding_instrument_type', ''),
                        match.get('award_ceiling', 0),
                        match['similarity_score']
                    )
                    
                    matches.append(match)
            
            # Sort by combined score and return top k
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matches[:k]
            
        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            raise

    def load_data_chunk(self, start_idx, chunk_size):
        """Load a specific chunk of data."""
        if not self.data_path:
            raise ValueError("Data path not set. Call load_model first.")
            
        chunk_data = []
        try:
            with open(self.data_path, 'rb') as f:
                # Skip to the start position
                for _ in range(start_idx):
                    try:
                        pickle.load(f)
                    except EOFError:
                        return []
                
                # Read the chunk
                for _ in range(chunk_size):
                    try:
                        item = pickle.load(f)
                        chunk_data.append(item)
                    except EOFError:
                        break
                    except Exception as e:
                        logger.warning(f"Error loading item: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading data chunk: {str(e)}")
            
        return chunk_data

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
    
    @lru_cache(maxsize=512)
    def _generate_justification(self, mission: str, grant_title: str, grant_description: str, 
                              grant_funding_type: str, award_ceiling: float, score: float) -> Dict:
        """Generate a human-readable justification for the match."""
        try:
            # Extract key terms from mission and grant
            mission_terms = self._extract_key_terms(mission)
            grant_terms = self._extract_key_terms(grant_title + " " + grant_description)
            
            # Find shared keywords
            shared_keywords = list(mission_terms & grant_terms)
            
            # Generate alignment summary
            alignment_summary = []
            if score >= 0.8:
                alignment_summary.append("Very strong alignment with organization's mission")
            elif score >= 0.6:
                alignment_summary.append("Good alignment with organization's mission")
            else:
                alignment_summary.append("Moderate alignment with organization's mission")
            
            # Add funding type alignment
            funding_alignment = []
            if grant_funding_type:
                funding_alignment.append(f"Grant type: {grant_funding_type}")
            if award_ceiling:
                funding_alignment.append(f"Maximum award: ${award_ceiling:,.2f}")
            
            return {
                'alignment_summary': alignment_summary,
                'funding_alignment': funding_alignment,
                'shared_keywords': shared_keywords[:10]  # Limit to top 10 keywords
            }
            
        except Exception as e:
            logger.error(f"Error generating justification: {str(e)}")
            return {
                'alignment_summary': ["Unable to generate detailed alignment analysis"],
                'funding_alignment': [],
                'shared_keywords': []
            }

    def get_recommendations(self, mission_statement: str, nonprofit_info: Dict, 
                            top_n: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Get grant recommendations for a nonprofit."""
        try:
            if not self.index:
                raise ValueError("Model not loaded. Call fit() or load_model() first.")
            
            # Encode mission statement
            query_vector = self.model.encode([mission_statement])[0]
            query_vector = query_vector.reshape(1, -1)
            
            # Get semantic matches using FAISS
            k = min(top_n * 3, len(self.grant_ids))  # Get more candidates for reranking
            distances, indices = self.index.search(query_vector, k)
            
            # Convert distances to similarities (1 - normalized distance)
            similarities = 1 - distances[0] / np.max(distances[0])
            
            # Get TF-IDF score for mission statement
            query_tfidf = self.tfidf.transform([mission_statement])
            
            # Compute TF-IDF similarities for candidates
            tfidf_similarities = []
            for idx in indices[0]:
                grant_tfidf = self.grant_tfidf_matrix[idx]
                similarity = (query_tfidf * grant_tfidf.T).toarray()[0][0]
                tfidf_similarities.append(similarity)
            
            # Combine scores (0.7 semantic + 0.3 TF-IDF)
            combined_scores = 0.7 * similarities + 0.3 * np.array(tfidf_similarities)
            
            # Get matches above threshold
            matches = []
            for idx, score in zip(indices[0], combined_scores):
                if score < min_similarity:
                    continue
                    
                grant = self.grants_data[idx].copy()
                grant['similarity_score'] = float(score)
                grant['original_similarity'] = float(score)
                
                # Generate match justification
                grant['match_justification'] = self._generate_justification(
                    mission_statement,
                    grant['opportunity_title'],
                    grant['grant_description'],
                    grant.get('funding_instrument_type', ''),
                    grant.get('award_ceiling', 0),
                    score
                )
                
                matches.append(grant)
            
            # Sort by score and return top_n
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matches[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    def load_data(self, data_path, chunk_size=1000):
        """Load model data in chunks to manage memory."""
        try:
            data = []
            with open(data_path, 'rb') as f:
                while True:
                    try:
                        chunk = pickle.load(f)
                        if isinstance(chunk, list):
                            data.extend(chunk)
                        else:
                            data.append(chunk)
                    except EOFError:
                        break
                    except MemoryError:
                        logger.warning("Memory limit reached while loading data")
                        if not data:
                            raise MemoryError("Not enough memory to load minimum required data")
                        break
            self.data = data
            logger.info(f"Successfully loaded {len(data)} records from {data_path}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise 

    def _init_funder_preferences(self):
        """Initialize funder preferences from grant data."""
        self.funder_preferences = {}
        for grant in self.data:
            agency = grant.get('agency', '')
            if agency:
                if agency not in self.funder_preferences:
                    self.funder_preferences[agency] = {
                        'total_grants': 0,
                        'avg_award': 0,
                        'categories': set(),
                        'funding_types': set()
                    }
                self.funder_preferences[agency]['total_grants'] += 1
                if 'award_ceiling' in grant and grant['award_ceiling']:
                    current_avg = self.funder_preferences[agency]['avg_award']
                    total = current_avg * (self.funder_preferences[agency]['total_grants'] - 1)
                    new_avg = (total + grant['award_ceiling']) / self.funder_preferences[agency]['total_grants']
                    self.funder_preferences[agency]['avg_award'] = new_avg
                if 'category' in grant:
                    self.funder_preferences[agency]['categories'].add(grant['category'])
                if 'funding_instrument_type' in grant:
                    self.funder_preferences[agency]['funding_types'].add(grant['funding_instrument_type']) 