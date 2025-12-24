
import base64
import io
import json  # For parsing Vision API responses
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from PIL import Image
import fitz  # PyMuPDF - Pure Python, no poppler needed!
from openai import OpenAI


@dataclass
class VisionExtractionResult:
    """Result from vision-based extraction"""
    parameter_id: str
    parameter_name: str
    value: Any
    source: str  # Specific section/location
    page_number: int
    confidence: float
    context: str  # Surrounding text/context


class VisionDocumentParser:
    
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o"):
       
        self.client = openai_client
        self.model = model
        self._image_cache = {}  # Cache converted images by PDF path
        logger.info(f"VisionDocumentParser initialized with model: {model}")
    
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        
        try:
            # Check cache first - ONLY OPTIMIZATION!
            cache_key = f"{pdf_path}_{dpi}"
            if cache_key in self._image_cache:
                logger.info(f"✅ Using CACHED images for: {Path(pdf_path).name} (skipping conversion)")
                return self._image_cache[cache_key]
            
            logger.info(f"Converting PDF to images: {Path(pdf_path).name} (DPI: {dpi})")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            images = []
            
            # Convert each page to image
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Calculate zoom factor for DPI
                # 72 DPI is default, so zoom = target_dpi / 72
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert pixmap to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                images.append(img)
            
            doc.close()
            
            # Cache for reuse - ONLY OPTIMIZATION!
            self._image_cache[cache_key] = images
            
            logger.success(f"Converted {len(images)} pages to images (PyMuPDF) - CACHED for reuse ✅")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            return []
    
    
    def image_to_base64(self, image: Image.Image) -> str:
        
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return ""
    
    
    def extract_all_parameters_from_page(
        self,
        image: Image.Image,
        page_num: int,
        parameters: List[Dict[str, str]]
    ) -> Dict[str, VisionExtractionResult]:
        
        try:
            # Build comprehensive prompt for ALL parameters
            param_descriptions = []
            for i, param in enumerate(parameters, 1):
                param_type = param.get('type', 'text')
                type_hint = {
                    'boolean': '(true/false)',
                    'number': '(numeric value)',
                    'date': '(date format)',
                    'text': '(text value)'
                }.get(param_type, '')
                
                param_descriptions.append(
                    f"{i}. **{param['name']}** {type_hint}: {param['description']}"
                )
            
            params_text = "\n".join(param_descriptions)
            
            prompt = f"""Analyze this document page and extract ALL of the following parameters that you can find:

{params_text}

IMPORTANT INSTRUCTIONS:
1. Return a JSON object with ONLY the parameters you found on this page
2. For each parameter found, provide:
   - "value": The actual value (use correct data type: number, boolean, string, or null)
   - "source": SPECIFIC location (e.g., "Account Summary Table - Settlement column, Row 2")
   - "confidence": Your confidence level (0.0 to 1.0)
   - "context": Brief surrounding text for verification

3. Skip parameters not visible on this page (don't include them in response)
4. Be precise with sources - include table names, section headers, row/column identifiers
5. For booleans, return true/false, NOT "yes"/"no" or 1/0

Return ONLY valid JSON, no markdown formatting:
{{
  "parameter_id_1": {{
    "found": true,
    "value": <actual_value>,
    "source": "Specific location with details",
    "confidence": 0.95,
    "context": "Surrounding text..."
  }},
  "parameter_id_2": {{
    "found": true,
    "value": <actual_value>,
    "source": "Another specific location",
    "confidence": 0.90,
    "context": "More context..."
  }}
}}

Parameter IDs to use: {', '.join([p['id'] for p in parameters])}"""

            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Single API call for ALL parameters!
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.0
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            results_dict = json.loads(content)
            
            # Create mapping of param_id to param_name for lookup
            param_name_map = {p['id']: p['name'] for p in parameters}
            
            # Convert to VisionExtractionResult objects
            extraction_results = {}
            for param_id, result_data in results_dict.items():
                if result_data.get('found', False):
                    extraction_results[param_id] = VisionExtractionResult(
                        parameter_id=param_id,
                        parameter_name=param_name_map.get(param_id, param_id),  # Get name from map
                        value=result_data.get('value'),
                        source=result_data.get('source', f'Page {page_num}'),
                        page_number=page_num,
                        confidence=result_data.get('confidence', 0.7),
                        context=result_data.get('context', '')
                    )
            
            logger.success(
                f"Page {page_num}: Found {len(extraction_results)}/{len(parameters)} parameters "
                f"in ONE call ⚡"
            )
            
            return extraction_results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from page {page_num}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting from page {page_num}: {str(e)}")
            return {}
    
    def extract_all_parameters_batch(
        self,
        pdf_path: str,
        parameters: List[Dict[str, str]]
    ) -> Dict[str, VisionExtractionResult]:
        
        try:
            logger.info(
                f"⚡ BATCH EXTRACTION: Processing {len(parameters)} parameters "
                f"from {Path(pdf_path).name}"
            )
            
            # Convert PDF to images (uses cache!)
            images = self.pdf_to_images(pdf_path, dpi=200)
            if not images:
                logger.error("Failed to convert PDF to images")
                return {}
            
            # Store best result for each parameter
            best_results = {}
            
            # Process each page once, extracting ALL parameters
            for page_num, image in enumerate(images, start=1):
                logger.info(f"⚡ Page {page_num}/{len(images)}: Extracting ALL parameters...")
                
                # Extract all parameters from this page in ONE call!
                page_results = self.extract_all_parameters_from_page(
                    image=image,
                    page_num=page_num,
                    parameters=parameters
                )
                
                # Update best results (keep highest confidence for each parameter)
                for param_id, result in page_results.items():
                    if param_id not in best_results:
                        best_results[param_id] = result
                        logger.info(f"  ✓ {param_id}: {result.value} (conf: {result.confidence})")
                    elif result.confidence > best_results[param_id].confidence:
                        logger.info(
                            f"  ↑ {param_id}: {result.value} (conf: {result.confidence}) "
                            f"[better than {best_results[param_id].confidence}]"
                        )
                        best_results[param_id] = result
            
            found_count = len(best_results)
            logger.success(
                f"⚡ BATCH COMPLETE: Found {found_count}/{len(parameters)} parameters "
                f"in {len(images)} API calls (vs {len(parameters) * len(images)} with old method!)"
            )
            
            return best_results
            
        except Exception as e:
            logger.error(f"Error in batch extraction: {str(e)}")
            return {}
    
    def extract_parameter_from_page(
        self,
        image: Image.Image,
        page_num: int,
        parameter_name: str,
        parameter_description: str,
        parameter_type: str = "text"
    ) -> Optional[VisionExtractionResult]:
        
        try:
            # Convert image to base64
            img_base64 = self.image_to_base64(image)
            if not img_base64:
                return None
            
            # Build prompt based on parameter type
            prompt = self._build_extraction_prompt(
                parameter_name,
                parameter_description,
                parameter_type
            )
            
            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.0  # Deterministic for data extraction
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            
            # Parse structured response
            return self._parse_vision_response(
                result_text,
                parameter_name,
                page_num
            )
            
        except Exception as e:
            logger.error(f"Error extracting {parameter_name} from page {page_num}: {str(e)}")
            return None
    
    
    def _build_extraction_prompt(
        self,
        parameter_name: str,
        parameter_description: str,
        parameter_type: str
    ) -> str:
        """Build prompt for GPT-4 Vision extraction"""
        
        prompt = f"""You are analyzing a financial document (Bureau Credit Report or GST Return).

**TASK:** Extract the following parameter from this document page.

**Parameter Name:** {parameter_name}
**Description:** {parameter_description}
**Expected Type:** {parameter_type}

**INSTRUCTIONS:**
1. Look for this parameter in the document
2. If found, extract the exact value
3. Note the specific section/location where you found it (e.g., "Account Summary Table, Row 3" or "DPD History Section")
4. Provide surrounding context (nearby text)

**OUTPUT FORMAT (JSON):**
{{
    "found": true/false,
    "value": <extracted value or null>,
    "source": "<specific section/table/location>",
    "confidence": <0.0-1.0>,
    "context": "<surrounding text for verification>"
}}

**EXAMPLES:**

For "DPD 30 Days" in a credit report:
{{
    "found": true,
    "value": 2,
    "source": "Payment History Table - DPD 30 Days column",
    "confidence": 0.95,
    "context": "DPD History: 0-30 days: 2 occurrences"
}}

For "Settlement/Write-off" flag:
{{
    "found": true,
    "value": false,
    "source": "Account Status Summary - Settlement Status field",
    "confidence": 0.90,
    "context": "Settlement Status: Not Applicable, Write-off Status: No"
}}

If parameter not found on this page:
{{
    "found": false,
    "value": null,
    "source": "Not found on this page",
    "confidence": 0.0,
    "context": ""
}}

**CRITICAL RULES:**
- Be precise with locations (section names, table names, row/column)
- Extract EXACT values, don't interpret
- For boolean parameters, return true/false
- For numeric parameters, return numbers (not strings)
- If unsure, set confidence < 0.7
- Return ONLY valid JSON, no other text

Now analyze the document image and extract the parameter:"""
        
        return prompt
    
    
    def _parse_vision_response(
        self,
        response_text: str,
        parameter_id: str,
        page_num: int
    ) -> Optional[VisionExtractionResult]:
        """Parse GPT-4 Vision response into structured result"""
        try:
            import json
            
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Check if found
            if not data.get("found", False):
                return None
            
            # Build result
            result = VisionExtractionResult(
                parameter_id=parameter_id,
                parameter_name=parameter_id.replace("_", " ").title(),
                value=data.get("value"),
                source=data.get("source", "Unknown location"),
                page_number=page_num,
                confidence=float(data.get("confidence", 0.5)),
                context=data.get("context", "")
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing vision response: {str(e)}")
            logger.debug(f"Response text: {response_text}")
            return None
    
    
    def extract_parameter_from_pdf(
        self,
        pdf_path: str,
        parameter_name: str,
        parameter_description: str,
        parameter_type: str = "text",
        search_all_pages: bool = True  # Search all pages for best accuracy
    ) -> Optional[VisionExtractionResult]:
        
        try:
            logger.info(f"Extracting '{parameter_name}' from {Path(pdf_path).name}")
            
            # Convert PDF to images (uses cache if already converted! - ONLY OPTIMIZATION)
            images = self.pdf_to_images(pdf_path, dpi=200)
            if not images:
                logger.error("Failed to convert PDF to images")
                return None
            
            # Search pages
            results = []
            
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Searching page {page_num}/{len(images)}...")
                
                result = self.extract_parameter_from_page(
                    image=image,
                    page_num=page_num,
                    parameter_name=parameter_name,
                    parameter_description=parameter_description,
                    parameter_type=parameter_type
                )
                
                if result and result.value is not None:
                    logger.success(f"Found on page {page_num}: {result.value} (confidence: {result.confidence})")
                    results.append(result)
                    
                    # Stop if we found a good match and not searching all pages
                    if not search_all_pages and result.confidence > 0.7:
                        break
            
            # Return best result
            if results:
                best_result = max(results, key=lambda r: r.confidence)
                logger.success(f"Best match: page {best_result.page_number}, confidence {best_result.confidence}")
                return best_result
            else:
                logger.warning(f"Parameter '{parameter_name}' not found in document")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting parameter from PDF: {str(e)}")
            return None
    
    
    def extract_gst_sales_with_vision(
        self,
        pdf_path: str
    ) -> Optional[Dict[str, Any]]:
        
        try:
            logger.info(f"Extracting GST sales from {Path(pdf_path).name}")
            
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path)
            if not images:
                return None
            
            # Prompt for GST sales
            prompt = """You are analyzing a GSTR-3B (GST Return) document.

**TASK:** Extract the total taxable sales value from Table 3.1(a).

**WHAT TO LOOK FOR:**
- Table 3.1(a): "Details of Outward Supplies and inward supplies liable to reverse charge"
- Look for "Taxable value" or "Total Taxable value"
- This is usually in the first row of Table 3.1

**OUTPUT FORMAT (JSON):**
{{
    "found": true/false,
    "month": "<month and year, e.g., January 2025>",
    "sales": <numeric value>,
    "source": "GSTR-3B Table 3.1(a)",
    "confidence": <0.0-1.0>
}}

**EXAMPLE:**
{{
    "found": true,
    "month": "January 2025",
    "sales": 951381,
    "source": "GSTR-3B Table 3.1(a) - Taxable outward supplies",
    "confidence": 0.95
}}

Return ONLY valid JSON, no other text."""
            
            # Try each page
            for page_num, image in enumerate(images, start=1):
                try:
                    img_base64 = self.image_to_base64(image)
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=300,
                        temperature=0.0
                    )
                    
                    result_text = response.choices[0].message.content
                    
                    # Parse JSON
                    import json
                    json_text = result_text.strip()
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```")[1].split("```")[0].strip()
                    
                    data = json.loads(json_text)
                    
                    if data.get("found") and data.get("sales"):
                        logger.success(f"Found GST sales on page {page_num}: {data['sales']}")
                        return {
                            "month": data.get("month", "Unknown"),
                            "sales": data["sales"],
                            "source": data.get("source", "GSTR-3B Table 3.1(a)")
                        }
                        
                except Exception as e:
                    logger.debug(f"Page {page_num} - no sales data: {str(e)}")
                    continue
            
            logger.warning("GST sales not found in document")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting GST sales: {str(e)}")
            return None



