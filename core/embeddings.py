from openai import OpenAI
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dataclasses import dataclass
import time


@dataclass
class SimilarityResult:
    """similarity search result"""
    index: int
    similarity: float
    text: str
    metadata: Dict[str, Any]


class EmbeddingService:
    """
    OpenAI embeddings service
    handles embedding creation and similarity search
    """

    def __init__(self, api_key, model="text-embedding-3-large"):
        try:
            self.client = OpenAI(api_key=api_key)
            self.model = model
            # dimensions based on model
            if "large" in model:
                self.embedding_dim = 1536
            else:
                self.embedding_dim = 1024
            self.max_tokens = 8191

            logger.info(f"EmbeddingService ready (model={model}, dims={self.embedding_dim})")

        except Exception as e:
            logger.error(f"Failed to init EmbeddingService: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def create_embedding(self, text):
        """create embedding for single text with retry"""
        try:
            # check if text is valid
            if not text or not text.strip():
                logger.warning("Empty text for embedding")
                return None

            # truncate long text (rough estimate: 4 chars per token)
            if len(text) > 30000:
                text = text[:30000]
                logger.warning("Text truncated to 30k chars")

            start_time = time.time()
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )

            embedding = response.data[0].embedding
            elapsed = time.time() - start_time

            logger.debug(f"Created embedding in {elapsed:.2f}s ({len(text)} chars → {len(embedding)} dims)")

            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def create_embeddings_batch(self, texts, batch_size=100):
        """create embeddings for multiple texts in batches"""
        try:
            if not texts:
                logger.warning("Empty text list for batch embedding")
                return []

            logger.info(f"Creating embeddings for {len(texts)} texts (batch_size={batch_size})")

            all_embeddings = []

            # process in chunks
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # clean up batch
                processed_batch = []
                for text in batch:
                    if text and text.strip():
                        # truncate if needed
                        if len(text) > 30000:
                            processed_batch.append(text[:30000])
                        else:
                            processed_batch.append(text)
                    else:
                        processed_batch.append(" ")  # fallback for empty

                try:
                    start_time = time.time()
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=processed_batch,
                        encoding_format="float"
                    )

                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)

                    elapsed = time.time() - start_time
                    batch_num = i//batch_size + 1
                    logger.debug(f"Batch {batch_num}: {len(batch)} texts in {elapsed:.2f}s")

                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    # add None for failed ones
                    for _ in range(len(batch)):
                        all_embeddings.append(None)

            # count successful embeddings
            successful = 0
            for e in all_embeddings:
                if e is not None:
                    successful += 1

            success_rate = (successful / len(texts)) * 100
            logger.success(f"Created {successful}/{len(texts)} embeddings ({success_rate:.1f}% success)")

            return all_embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            return None

    def cosine_similarity(self, vec1, vec2):
        """calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            # handle zero vectors
            if norm1 == 0 or norm2 == 0:
                logger.warning("Zero vector in cosine similarity")
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # clip to valid range (sometimes gets > 1 due to numerical errors)
            similarity = float(np.clip(similarity, 0.0, 1.0))

            return similarity

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    def find_most_similar(self, query_embedding, candidate_embeddings, 
                         candidate_texts=None, candidate_metadata=None,
                         top_k=5, min_similarity=0.0):
        """
        find most similar embeddings using cosine similarity
        returns sorted list of results
        """
        try:
            if not query_embedding or not candidate_embeddings:
                logger.warning("Empty embeddings for similarity search")
                return []

            logger.info(f"Finding top {top_k} from {len(candidate_embeddings)} candidates (min={min_similarity})")

            similarities = []

            for idx, candidate in enumerate(candidate_embeddings):
                # skip None embeddings
                if candidate is None:
                    continue

                try:
                    similarity = self.cosine_similarity(query_embedding, candidate)

                    # filter by threshold
                    if similarity >= min_similarity:
                        text = ""
                        if candidate_texts:
                            text = candidate_texts[idx]

                        metadata = {}
                        if candidate_metadata:
                            metadata = candidate_metadata[idx]

                        result = SimilarityResult(
                            index=idx,
                            similarity=similarity,
                            text=text,
                            metadata=metadata
                        )
                        similarities.append(result)

                except Exception as e:
                    logger.warning(f"Error computing similarity for idx {idx}: {str(e)}")
                    continue

            # sort by similarity descending
            similarities.sort(key=lambda x: x.similarity, reverse=True)

            # get top k
            top_results = similarities[:top_k]

            if top_results:
                logger.success(f"Found {len(top_results)} results (top: {top_results[0].similarity:.3f})")
            else:
                logger.warning("No results above threshold")

            return top_results

        except Exception as e:
            logger.error(f"Error in find_most_similar: {str(e)}")
            return []

    def embed_documents(self, texts, metadata=None):
        """embed multiple documents with metadata"""
        try:
            logger.info(f"Embedding {len(texts)} documents")

            embeddings = self.create_embeddings_batch(texts)

            if embeddings is None:
                logger.error("Batch embedding failed")
                return [], []

            # create metadata if missing
            if metadata is None:
                metadata = []
                for _ in texts:
                    metadata.append({})

            # add embedding info to metadata
            for i in range(len(embeddings)):
                embedding = embeddings[i]
                meta = metadata[i]

                if embedding is not None:
                    meta["embedding_dim"] = len(embedding)
                    meta["has_embedding"] = True
                else:
                    meta["has_embedding"] = False

            return embeddings, metadata

        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            return [], []

    def get_embedding_stats(self, embeddings):
        """get stats about embeddings"""
        try:
            valid_embeddings = []
            for e in embeddings:
                if e is not None:
                    valid_embeddings.append(e)

            if not valid_embeddings:
                return {
                    "total": len(embeddings),
                    "valid": 0,
                    "invalid": len(embeddings),
                    "success_rate": 0.0
                }

            # convert to numpy for calculations
            emb_array = np.array(valid_embeddings)

            # calculate norms
            norms = []
            for e in valid_embeddings:
                norms.append(np.linalg.norm(e))

            stats = {
                "total": len(embeddings),
                "valid": len(valid_embeddings),
                "invalid": len(embeddings) - len(valid_embeddings),
                "success_rate": len(valid_embeddings) / len(embeddings),
                "dimensions": len(valid_embeddings[0]) if valid_embeddings else 0,
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms))
            }

            logger.info(f"Embedding stats: {stats['valid']}/{stats['total']} valid ({stats['success_rate']*100:.1f}%)")

            return stats

        except Exception as e:
            logger.error(f"Error calculating stats: {str(e)}")
            return {}
