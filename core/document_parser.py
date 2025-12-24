

import pdfplumber
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class DocumentChunk:
    """chunk of text from document"""
    chunk_id: str
    text: str
    page_num: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


@dataclass
class ParsedDocument:
    """parsed document data"""
    file_name: str
    total_pages: int
    text_content: str
    pages: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]


class DocumentParser:
    # PDF parser with chunking for RAG

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Parser initialized - chunk_size={chunk_size}, overlap={chunk_overlap}")

    def parse_pdf(self, pdf_path):
        """
        parse PDF and extract content
        """
        logger.info(f"Parsing: {Path(pdf_path).name}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = []
                pages_data = []
                tables_data = []

                # go through each page
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_result = self._parse_page(page, page_num)

                        all_text.append(page_result["text"])
                        pages_data.append(page_result["page_data"])
                        tables_data.extend(page_result["tables"])

                        logger.debug(f"Page {page_num}: {len(page_result['text'])} chars, {len(page_result['tables'])} tables")

                    except Exception as e:
                        logger.error(f"Error on page {page_num}: {str(e)}")
                        continue  # skip problematic pages

                full_text = "\n\n".join(all_text)

                # create chunks for embeddings
                chunks = self._create_chunks(full_text, Path(pdf_path).name)

                metadata = {
                    "file_path": pdf_path,
                    "file_name": Path(pdf_path).name,
                    "total_pages": len(pdf.pages),
                    "total_tables": len(tables_data),
                    "total_chunks": len(chunks),
                    "text_length": len(full_text)
                }

                parsed_doc = ParsedDocument(
                    file_name=Path(pdf_path).name,
                    total_pages=len(pdf.pages),
                    text_content=full_text,
                    pages=pages_data,
                    tables=tables_data,
                    chunks=chunks,
                    metadata=metadata
                )

                logger.success(f"Parsed {len(pdf.pages)} pages, {len(tables_data)} tables, {len(chunks)} chunks")

                return parsed_doc

        except FileNotFoundError:
            logger.error(f"File not found: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {str(e)}")
            return None

    def _parse_page(self, page, page_num):
        """parse single page"""
        try:
            # grab text
            page_text = page.extract_text()
            if page_text is None:
                page_text = ""

            # extract tables
            tables = []
            raw_tables = page.extract_tables()

            for table_idx, table in enumerate(raw_tables):
                if table and len(table) > 0:
                    try:
                        table_data = {
                            "page": page_num,
                            "table_id": f"p{page_num}_t{table_idx + 1}",
                            "headers": table[0] if table else [],
                            "rows": table[1:] if len(table) > 1 else [],
                            "raw_data": table
                        }
                        tables.append(table_data)
                    except Exception as e:
                        logger.warning(f"Table {table_idx} error on page {page_num}: {str(e)}")

            page_data = {
                "page_num": page_num,
                "text": page_text,
                "text_length": len(page_text),
                "tables_count": len(tables),
                "width": page.width,
                "height": page.height
            }

            return {
                "text": page_text,
                "tables": tables,
                "page_data": page_data
            }

        except Exception as e:
            logger.error(f"_parse_page error for page {page_num}: {str(e)}")
            return {
                "text": "",
                "tables": [],
                "page_data": {
                    "page_num": page_num,
                    "text": "",
                    "text_length": 0,
                    "tables_count": 0
                }
            }

    def _create_chunks(self, text, file_name):
        """
        break text into chunks with overlap
        TODO: maybe improve the chunking logic later
        """
        try:
            chunks = []

            if not text:
                logger.warning("Empty text for chunking")
                return chunks

            # split by paragraphs
            paragraphs = text.split('\n\n')

            current_chunk = ""
            current_start = 0
            chunk_id = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # check if adding para exceeds size
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    # save chunk
                    chunk = DocumentChunk(
                        chunk_id=f"chunk_{chunk_id}",
                        text=current_chunk.strip(),
                        page_num=0,  # not tracking page num for now
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={
                            "source_file": file_name,
                            "chunk_length": len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                    # start new chunk with overlap
                    if len(current_chunk) > self.chunk_overlap:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                    else:
                        overlap_text = current_chunk
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    # add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

            # add final chunk
            if current_chunk:
                chunk = DocumentChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=current_chunk.strip(),
                    page_num=0,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        "source_file": file_name,
                        "chunk_length": len(current_chunk)
                    }
                )
                chunks.append(chunk)

            logger.info(f"Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            return []

    def extract_bureau_score(self, parsed_doc):
        """
        grab CIBIL score from CRIF report
        looks for pattern like "PERFORM CONSUMER 2.2 300-900 627"
        """
        try:
            text = parsed_doc.text_content

            # main pattern - score after range
            pattern = r'PERFORM\s+CONSUMER.*?300-900\s+(\d{3})'
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                score = int(match.group(1))
                if 300 <= score <= 900:
                    logger.info(f"Found bureau score: {score}")
                    return {
                        "value": score,
                        "source": "CRIF Report – Score Section"
                    }

            # fallback - check first couple pages
            for page in parsed_doc.pages[:2]:
                page_text = page["text"]
                numbers = re.findall(r'\b(\d{3})\b', page_text)

                for num_str in numbers:
                    num = int(num_str)
                    if 300 <= num <= 900:
                        # check if its actually a score
                        idx = page_text.find(num_str)
                        context = page_text[max(0, idx-100):idx+100]

                        keywords = ['score', 'cibil', 'credit', 'bureau']
                        if any(kw in context.lower() for kw in keywords):
                            logger.info(f"Found score (fallback): {num}")
                            return {
                                "value": num,
                                "source": f"CRIF Report – Page {page['page_num']}"
                            }

            logger.warning("Bureau score not found")
            return None

        except Exception as e:
            logger.error(f"Error extracting bureau score: {str(e)}")
            return None

    def extract_gst_sales(self, parsed_doc):
        """extract sales from GSTR-3B table"""
        try:
            text = parsed_doc.text_content
            filename = parsed_doc.file_name

            # get month from document
            month_match = re.search(r'Period\s+(\w+)', text)
            month_name = month_match.group(1) if month_match else "Unknown"

            # extract year from filename (GSTR3B_..._012025.pdf format)
            filename_year_match = re.search(r'_(\d{2})(\d{4})\.pdf', filename)
            if filename_year_match:
                year = filename_year_match.group(2)
            else:
                # fallback
                year_match = re.search(r'Year\s+(\d{4})', text)
                year = year_match.group(1) if year_match else "2025"

            formatted_month = f"{month_name} {year}"

            # search tables for sales
            for table in parsed_doc.tables:
                rows = table.get("rows", [])

                for row in rows:
                    if row and len(row) > 1:
                        first_cell = str(row[0]).replace('\n', ' ')

                        # find row (a) with outward supplies
                        if "(a)" in first_cell and "Outward taxable supplies" in first_cell:
                            if len(row) > 1 and row[1]:
                                value_str = str(row[1])
                                clean_value = re.sub(r'[^\d.]', '', value_str)

                                if clean_value:
                                    try:
                                        sales = float(clean_value)
                                        logger.info(f"GST sales: {sales} for {formatted_month}")
                                        return {
                                            "month": formatted_month,
                                            "sales": sales,
                                            "source": "GSTR-3B Table 3.1(a)"
                                        }
                                    except ValueError as e:
                                        logger.warning(f"Couldn't parse sales value '{clean_value}': {str(e)}")

            logger.warning(f"Sales data not found for {formatted_month}")
            return None

        except Exception as e:
            logger.error(f"Error extracting GST sales: {str(e)}")
            return None

    def get_chunks_text(self, chunks):
        """get text from chunks for embedding"""
        try:
            return [chunk.text for chunk in chunks]
        except Exception as e:
            logger.error(f"Error getting chunks text: {str(e)}")
            return []
