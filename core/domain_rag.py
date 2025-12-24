from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger

from .domain_knowledge import ALL_KNOWLEDGE, KNOWLEDGE_CATEGORIES
from .embeddings import EmbeddingService


@dataclass
class DomainSnippet:
    """domain knowledge snippet with its embedding"""
    text: str
    embedding: np.ndarray
    category: str
    index: int


class DomainRAG:
    """
    RAG for domain knowledge
    embeds hardcoded domain notes on startup
    """

    def __init__(self, embedding_service):
        try:
            self.embedding_service = embedding_service
            self.snippets = []
            self._initialize_knowledge()

        except Exception as e:
            logger.error(f"Failed to init DomainRAG: {str(e)}")
            raise

    def _initialize_knowledge(self):
        """embed all domain knowledge on startup"""
        try:
            logger.info("Initializing domain knowledge RAG...")
            logger.info(f"Total snippets to embed: {len(ALL_KNOWLEDGE)}")

            # batch embed all snippets
            embeddings = self.embedding_service.create_embeddings_batch(
                texts=ALL_KNOWLEDGE
            )

            if not embeddings or len(embeddings) != len(ALL_KNOWLEDGE):
                logger.error("Failed to create embeddings for domain knowledge")
                return

            # store snippets with their embeddings
            for i, text in enumerate(ALL_KNOWLEDGE):
                category = self._categorize_snippet(text)
                self.snippets.append(DomainSnippet(
                    text=text,
                    embedding=embeddings[i],
                    category=category,
                    index=i
                ))

            logger.success(f"Domain RAG ready: {len(self.snippets)} snippets embedded")

            # log category breakdown
            category_counts = {}
            for snippet in self.snippets:
                if snippet.category in category_counts:
                    category_counts[snippet.category] += 1
                else:
                    category_counts[snippet.category] = 1

            for category, count in category_counts.items():
                logger.info(f"  - {category}: {count} snippets")

        except Exception as e:
            logger.error(f"Error initializing domain knowledge: {str(e)}")
            # don't crash - system can work without it
            self.snippets = []

    def _categorize_snippet(self, text):
        """figure out what category this snippet belongs to"""
        try:
            text_lower = text.lower()

            # bureau stuff
            bureau_keywords = ['bureau', 'credit', 'dpd', 'score', 'cibil', 'loan', 
                             'settlement', 'write-off', 'ntc', 'suit']
            if any(kw in text_lower for kw in bureau_keywords):
                return "bureau"

            # gst related
            gst_keywords = ['gst', 'gstr', 'table', 'supply', 'taxable', 'outward',
                          'gstin', 'tax period']
            if any(kw in text_lower for kw in gst_keywords):
                return "gst"

            # validation rules
            validation_keywords = ['valid', 'should', 'must', 'rule', 'between', 'range',
                                 'validation', 'suspicious', 'negative']
            if any(kw in text_lower for kw in validation_keywords):
                return "validation"

            # extraction hints
            hint_keywords = ['location', 'found', 'appears', 'extraction', 'look for',
                           'typically', 'usually', 'marked']
            if any(kw in text_lower for kw in hint_keywords):
                return "extraction_hint"

            # pattern stuff
            pattern_keywords = ['format', 'pattern', 'representation', 'shown as',
                              'display', 'code', 'type']
            if any(kw in text_lower for kw in pattern_keywords):
                return "common_pattern"

            return "general"

        except Exception as e:
            logger.error(f"Error categorizing snippet: {str(e)}")
            return "general"

    def retrieve(self, query, top_k=3, min_similarity=0.3, category_filter=None):
        """
        get most relevant domain snippets for query
        """
        try:
            if not self.snippets:
                logger.warning("No domain knowledge available")
                return []

            logger.info(f"Retrieving domain knowledge for: '{query[:100]}...'")

            # embed the query
            query_embedding = self.embedding_service.create_embedding(query)

            if query_embedding is None:
                logger.error("Failed to create query embedding")
                return []

            # filter by category if needed
            filtered_snippets = self.snippets
            if category_filter:
                filtered_snippets = [s for s in self.snippets if s.category == category_filter]
                logger.info(f"Filtered to {len(filtered_snippets)} snippets in '{category_filter}'")

            if not filtered_snippets:
                logger.warning(f"No snippets for category: {category_filter}")
                return []

            # prepare data for similarity search
            snippet_embeddings = [s.embedding for s in filtered_snippets]
            snippet_texts = [s.text for s in filtered_snippets]
            snippet_metadata = [{"category": s.category, "index": s.index} for s in filtered_snippets]

            # find similar snippets
            results = self.embedding_service.find_most_similar(
                query_embedding=query_embedding,
                candidate_embeddings=snippet_embeddings,
                candidate_texts=snippet_texts,
                candidate_metadata=snippet_metadata,
                top_k=top_k,
                min_similarity=min_similarity
            )

            if results:
                logger.success(f"Retrieved {len(results)} snippets (top: {results[0].similarity:.3f})")
                return [r.text for r in results]
            else:
                logger.warning(f"No snippets above threshold {min_similarity}")
                return []

        except Exception as e:
            logger.error(f"Error retrieving domain knowledge: {str(e)}")
            return []

    def retrieve_by_category(self, query, categories, snippets_per_category=2):
        """get snippets grouped by category"""
        try:
            results = {}

            for category in categories:
                snippets = self.retrieve(
                    query=query,
                    top_k=snippets_per_category,
                    category_filter=category
                )
                if snippets:
                    results[category] = snippets

            return results

        except Exception as e:
            logger.error(f"Error retrieving by category: {str(e)}")
            return {}

    def get_all_snippets(self, category=None):
        """get all snippets, optionally filtered"""
        try:
            if category:
                return [s.text for s in self.snippets if s.category == category]
            else:
                return [s.text for s in self.snippets]

        except Exception as e:
            logger.error(f"Error getting snippets: {str(e)}")
            return []

    def get_statistics(self):
        """stats about domain knowledge"""
        try:
            category_counts = {}
            for snippet in self.snippets:
                if snippet.category in category_counts:
                    category_counts[snippet.category] += 1
                else:
                    category_counts[snippet.category] = 1

            embedding_dim = 0
            if self.snippets and len(self.snippets) > 0:
                embedding_dim = len(self.snippets[0].embedding)

            return {
                "total_snippets": len(self.snippets),
                "categories": category_counts,
                "embedding_dimension": embedding_dim
            }

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"total_snippets": 0, "categories": {}, "embedding_dimension": 0}
