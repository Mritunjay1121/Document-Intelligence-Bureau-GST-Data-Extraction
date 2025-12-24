from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger
from openai import OpenAI

from core.document_parser import ParsedDocument
from core.embeddings import EmbeddingService
from core.vision_parser import VisionDocumentParser, VisionExtractionResult


@dataclass
class ExtractionResult:
    """result from parameter extraction"""
    parameter_id: str
    parameter_name: str
    value: Any
    source: str
    confidence: float
    context_used: str
    metadata: Dict[str, Any]


class EnhancedRAGPipeline:
    """
    RAG Pipeline with Vision support
    tries vision first, falls back to traditional RAG
    """

    def __init__(self, embedding_service, openai_client, domain_rag=None, 
                 top_k=5, similarity_threshold=0.3, model="gpt-4o-mini",
                 vision_model="gpt-4o", temperature=0.0, use_vision=True):
        self.embedding_service = embedding_service
        self.client = openai_client
        self.domain_rag = domain_rag
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.temperature = temperature
        self.use_vision = use_vision

        # setup vision parser if needed
        if use_vision:
            self.vision_parser = VisionDocumentParser(
                openai_client=openai_client,
                model=vision_model
            )
            logger.success("Vision parser ready")
        else:
            self.vision_parser = None
            logger.info("Vision extraction disabled")

    def extract_parameter_with_vision(self, pdf_path, parameter_id, 
                                     parameter_name, parameter_description):
        """
        extract using GPT-4 Vision (most accurate method)
        """
        if not self.use_vision or not self.vision_parser:
            return None

        try:
            logger.info(f"[VISION] Extracting {parameter_name}...")

            # figure out what type of parameter this is
            param_type = self._infer_parameter_type(parameter_id, parameter_description)

            # use vision to extract
            vision_result = self.vision_parser.extract_parameter_from_pdf(
                pdf_path=pdf_path,
                parameter_name=parameter_name,
                parameter_description=parameter_description,
                parameter_type=param_type,
                search_all_pages=True
            )

            if vision_result:
                # convert to our format
                result = ExtractionResult(
                    parameter_id=parameter_id,
                    parameter_name=parameter_name,
                    value=vision_result.value,
                    source=vision_result.source,
                    confidence=vision_result.confidence,
                    context_used=vision_result.context,
                    metadata={
                        "method": "vision",
                        "page_number": vision_result.page_number,
                        "model": self.vision_parser.model
                    }
                )

                logger.success(f"[VISION] Found: {result.value} (conf: {result.confidence:.2f})")
                return result
            else:
                logger.warning(f"[VISION] Not found: {parameter_name}")
                return None

        except Exception as e:
            logger.error(f"[VISION] Error: {str(e)}")
            return None

    def _infer_parameter_type(self, parameter_id, description):
        """guess parameter type from id and description"""
        param_lower = parameter_id.lower()
        desc_lower = description.lower()

        # boolean stuff
        boolean_keywords = ["accepted", "flag", "status", "yes/no", "true/false", 
                          "settlement", "writeoff", "suit", "default"]
        for keyword in boolean_keywords:
            if keyword in param_lower or keyword in desc_lower:
                return "boolean"

        # numeric stuff
        numeric_keywords = ["amount", "count", "number", "dpd", "loans", 
                          "threshold", "score", "inquiries"]
        for keyword in numeric_keywords:
            if keyword in param_lower or keyword in desc_lower:
                return "number"

        # dates
        if "date" in param_lower or "date" in desc_lower:
            return "date"

        return "text"

    def prepare_document(self, parsed_doc):
        """
        prepare doc for traditional RAG (fallback)
        """
        try:
            chunk_texts = []
            chunk_metadata = []

            for chunk in parsed_doc.chunks:
                chunk_texts.append(chunk.text)
                chunk_metadata.append({
                    "chunk_id": chunk.chunk_id,
                    "page_num": chunk.page_num,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                })

            # create embeddings
            embeddings_list = self.embedding_service.create_embeddings_batch(chunk_texts)

            if embeddings_list is None:
                logger.error("Failed to create embeddings")
                return None, None, None

            # convert to numpy
            embeddings = np.array(embeddings_list)

            return embeddings, chunk_texts, chunk_metadata

        except Exception as e:
            logger.error(f"Error preparing document: {str(e)}")
            return None, None, None

    def extract_parameter_full_pipeline(self, parameter_id, parameter_name, 
                                       parameter_description, parsed_doc, 
                                       chunk_embeddings, chunk_texts, 
                                       chunk_metadata, pdf_path=None):
        """
        full extraction pipeline
        tries vision first, then RAG as fallback
        """
        try:
            # try vision first (best accuracy)
            if pdf_path and self.use_vision:
                vision_result = self.extract_parameter_with_vision(
                    pdf_path=pdf_path,
                    parameter_id=parameter_id,
                    parameter_name=parameter_name,
                    parameter_description=parameter_description
                )

                # if vision found it with good confidence, use that
                if vision_result and vision_result.confidence >= 0.7:
                    logger.success(f"[PIPELINE] Using VISION result (conf: {vision_result.confidence:.2f})")
                    return vision_result

            # try traditional RAG
            logger.info(f"[PIPELINE] Trying traditional RAG for {parameter_name}...")
            rag_result = self._extract_with_rag(
                parameter_id=parameter_id,
                parameter_name=parameter_name,
                parameter_description=parameter_description,
                chunk_embeddings=chunk_embeddings,
                chunk_texts=chunk_texts,
                chunk_metadata=chunk_metadata,
                parsed_doc=parsed_doc
            )

            # if we have both, compare them
            if vision_result and rag_result:
                if vision_result.confidence > rag_result.confidence:
                    logger.info(f"[PIPELINE] Vision wins: {vision_result.confidence:.2f} > {rag_result.confidence:.2f}")
                    return vision_result
                else:
                    logger.info(f"[PIPELINE] RAG wins: {rag_result.confidence:.2f} > {vision_result.confidence:.2f}")
                    return rag_result

            # return whatever worked
            if vision_result:
                return vision_result
            return rag_result

        except Exception as e:
            logger.error(f"Error in extraction pipeline: {str(e)}")
            return None

    def _extract_with_rag(self, parameter_id, parameter_name, parameter_description,
                         chunk_embeddings, chunk_texts, chunk_metadata, parsed_doc):
        """traditional RAG extraction (fallback method)"""
        try:
            # build query
            query = f"{parameter_name}: {parameter_description}"
            query_embedding = self.embedding_service.create_embedding(query)

            if query_embedding is None:
                return None

            # get relevant chunks
            similarities = np.dot(chunk_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:self.top_k]

            # filter by threshold
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    relevant_chunks.append({
                        "text": chunk_texts[idx],
                        "similarity": float(similarities[idx]),
                        "metadata": chunk_metadata[idx]
                    })

            if not relevant_chunks:
                return None

            # get domain knowledge if available
            domain_context = ""
            if self.domain_rag:
                domain_snippets = self.domain_rag.retrieve(query, top_k=3)
                if domain_snippets:
                    formatted_snippets = []
                    for s in domain_snippets:
                        # handle different types
                        if isinstance(s, str):
                            formatted_snippets.append(f"- {s}")
                        elif hasattr(s, 'text'):
                            formatted_snippets.append(f"- {s.text}")
                        else:
                            formatted_snippets.append(f"- {str(s)}")
                    domain_context = "\n".join(formatted_snippets)

            # build context from chunks
            context_parts = []
            for i, c in enumerate(relevant_chunks):
                chunk_text = f"[Chunk {i+1}, Page {c['metadata']['page_num']}, Similarity: {c['similarity']:.2f}]\n{c['text']}"
                context_parts.append(chunk_text)
            context_text = "\n\n".join(context_parts)

            # build prompt
            prompt = f"""Extract the following parameter from the document context.

Parameter: {parameter_name}
Description: {parameter_description}

"""
            if domain_context:
                prompt += f"Domain Knowledge:\n{domain_context}\n\n"

            prompt += f"""Document Context:
{context_text}

Extract the value and provide the specific source location (e.g., "Account Summary Table, Row 3" not just the filename).

Return JSON:
{{
    "value": <extracted value or null>,
    "source": "<specific section/table/location>",
    "confidence": <0.0-1.0>
}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=300
            )

            result_text = response.choices[0].message.content

            # parse JSON response
            import json
            json_text = result_text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            data = json.loads(json_text)

            if data.get("value") is not None:
                return ExtractionResult(
                    parameter_id=parameter_id,
                    parameter_name=parameter_name,
                    value=data["value"],
                    source=data.get("source", f"Page {relevant_chunks[0]['metadata']['page_num']}"),
                    confidence=float(data.get("confidence", 0.5)),
                    context_used=context_text[:200],
                    metadata={"method": "rag", "chunks_used": len(relevant_chunks)}
                )

            return None

        except Exception as e:
            logger.error(f"RAG extraction error: {str(e)}")
            return None

    def calculate_overall_confidence(self, results):
        """calculate overall confidence score"""
        if not results:
            return 0.0

        # count successful extractions
        successful = []
        for r in results:
            if r.value is not None:
                successful.append(r)

        if not successful:
            return 0.0

        # average confidence
        total_conf = 0.0
        for r in successful:
            total_conf += r.confidence
        avg_confidence = total_conf / len(successful)

        # success rate
        success_rate = len(successful) / len(results)

        # combine them
        overall = (avg_confidence * 0.7) + (success_rate * 0.3)

        return round(overall, 2)
