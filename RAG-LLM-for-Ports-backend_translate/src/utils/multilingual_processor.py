# src/utils/multilingual_processor.py
"""
Enhanced Multilingual Document Processing for AI Port Decision-Support System
Implements comprehensive text extraction, language detection, translation, and metadata preservation.
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# OCR libraries
try:
    import pytesseract
    import cv2
    import numpy as np
    from PIL import Image
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Language detection + translation
import langdetect
from langdetect.lang_detect_exception import LangDetectException
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------
# ENUMS + DATA CLASSES
# -------------------------------

class DocumentType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    SCANNED_PDF = "scanned_pdf"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    is_reliable: bool
    fallback_used: bool = False


@dataclass
class ProcessingMetadata:
    document_type: DocumentType
    original_language: str
    detected_language: str
    confidence_score: float
    is_translated: bool
    translation_model: Optional[str]
    ocr_used: bool
    ocr_confidence: Optional[float]
    processing_time: float
    document_hash: str
    file_size: int
    page_count: int
    chunk_count: int


# -------------------------------
# MAIN PROCESSOR
# -------------------------------

class EnhancedMultilingualProcessor:
    """
    Multilingual processor with OCR fallback, translation, and metadata preservation.
    """

    def __init__(
        self,
        supported_languages: List[str] = None,
        translation_model_name: str = "Helsinki-NLP/opus-mt-en-mul",
        enable_ocr: bool = True,
        ocr_languages: List[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.supported_languages = supported_languages or [
            "en", "es", "fr", "de", "zh", "ja", "ko", "ar", "ru", "pt"
        ]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_languages = ocr_languages or ["en", "es", "fr", "de", "zh", "ja", "ko"]

        self.language_detector = self._initialize_language_detector()
        self.translation_models = self._initialize_translation_models(translation_model_name)

        if self.enable_ocr:
            self.ocr_reader = self._initialize_ocr()
        else:
            self.ocr_reader = None
            logger.warning("OCR disabled or unavailable.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(f"EnhancedMultilingualProcessor initialized. OCR Enabled: {self.enable_ocr}")

    # -------------------------------------------------------------------------
    # MODELS
    # -------------------------------------------------------------------------

    def _initialize_language_detector(self):
        try:
            detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                return_all_scores=True,
            )
            return {"model": detector, "type": "advanced"}
        except Exception as e:
            logger.warning(f"Failed to load advanced language detector: {e}. Using fallback.")
            return {"model": None, "type": "fallback"}

    def _initialize_translation_models(self, main_model):
        models = {}

        # Check PyTorch version constraints
        try:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if major < 2 or (major == 2 and minor < 6):
                logger.warning(f"PyTorch {torch.__version__} is outdated for translation models.")
        except Exception:
            pass

        # Load main model
        try:
            models["main"] = {
                "model": MarianMTModel.from_pretrained(main_model),
                "tokenizer": MarianTokenizer.from_pretrained(main_model),
                "name": main_model,
            }
        except Exception as e:
            logger.warning(f"Could not load main translation model {main_model}: {e}")
            models["main"] = None

        # Additional pairs
        additional = {
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "en-de": "Helsinki-NLP/opus-mt-en-de",
            "en-zh": "Helsinki-NLP/opus-mt-en-zho",
            "en-ja": "Helsinki-NLP/opus-mt-en-jap",
        }

        for key, model_name in additional.items():
            try:
                models[key] = {
                    "model": MarianMTModel.from_pretrained(model_name),
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "name": model_name,
                }
            except Exception as e:
                logger.warning(f"Translation model {key} failed: {e}")
                models[key] = None

        return models

    def _initialize_ocr(self):
        try:
            reader = easyocr.Reader(self.ocr_languages, gpu=False)
            return reader
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            return None

    # -------------------------------------------------------------------------
    # DOCUMENT TYPE DETECTION (TEST-COMPATIBLE)
    # -------------------------------------------------------------------------

    def detect_document_type(self, file_path: str) -> DocumentType:
        """
        EXTENSION-ONLY detection (required by tests)
        """
        try:
            suffix = Path(file_path).suffix.lower()

            if suffix == ".pdf":
                return DocumentType.PDF
            if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                return DocumentType.IMAGE
            if suffix in [".txt", ".md", ".rtf"]:
                return DocumentType.TEXT

            return DocumentType.UNKNOWN

        except Exception as e:
            logger.warning(f"Document type detection failed for {file_path}: {e}")
            return DocumentType.UNKNOWN

    # -------------------------------------------------------------------------
    # TEXT EXTRACTION (FULL SCANNED-PDF SUPPORT)
    # -------------------------------------------------------------------------

    def extract_text_from_document(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        doc_type = self.detect_document_type(file_path)
        metadata = {
            "document_type": doc_type.value,
            "file_path": file_path,
            "extraction_method": None,
            "ocr_confidence": None,
            "processing_time": 0,
        }

        start = datetime.now()

        # Missing file → tests require "error"
        if not os.path.exists(file_path):
            metadata["error"] = f"File not found: {file_path}"
            metadata["processing_time"] = (datetime.now() - start).total_seconds()
            return "", metadata

        try:
            # -------------------------
            # PDF → try text → fallback OCR
            # -------------------------
            if doc_type == DocumentType.PDF:
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    text = "\n".join(doc.page_content for doc in docs).strip()

                    if text:
                        metadata["extraction_method"] = "pdf_text"
                    else:
                        # PDF but no text → scanned
                        if self.enable_ocr:
                            text, conf = self._extract_text_from_pdf_with_ocr(file_path)
                            metadata["extraction_method"] = "ocr"
                            metadata["ocr_confidence"] = conf
                            metadata["document_type"] = DocumentType.SCANNED_PDF.value
                        else:
                            text = ""
                except Exception:
                    # PyPDFLoader failed → scanned
                    if self.enable_ocr:
                        text, conf = self._extract_text_from_pdf_with_ocr(file_path)
                        metadata["extraction_method"] = "ocr"
                        metadata["ocr_confidence"] = conf
                        metadata["document_type"] = DocumentType.SCANNED_PDF.value
                    else:
                        text = ""

            # -------------------------
            # IMAGE or Explicit scanned
            # -------------------------
            elif doc_type in (DocumentType.IMAGE, DocumentType.SCANNED_PDF):
                if self.enable_ocr:
                    text, conf = self._extract_text_with_ocr(file_path)
                    metadata["extraction_method"] = "ocr"
                    metadata["ocr_confidence"] = conf
                else:
                    text = ""

            # -------------------------
            # PLAIN TEXT
            # -------------------------
            elif doc_type == DocumentType.TEXT:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                metadata["extraction_method"] = "text_file"

            else:
                text = ""

            metadata["processing_time"] = (datetime.now() - start).total_seconds()
            return text, metadata

        except Exception as e:
            metadata["error"] = str(e)
            metadata["processing_time"] = (datetime.now() - start).total_seconds()
            return "", metadata

    # -------------------------------------------------------------------------
    # OCR IMPLEMENTATION
    # -------------------------------------------------------------------------

    def _extract_text_with_ocr(self, file_path: str):
        try:
            img = cv2.imread(file_path)
            if img is None:
                return "", 0.0

            results = self.ocr_reader.readtext(img)
            texts, confs = [], []

            for (_, text, conf) in results:
                if conf > 0.3:
                    texts.append(text)
                    confs.append(conf)

            return "\n".join(texts), (sum(confs) / len(confs) if confs else 0.0)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return "", 0.0

    def _extract_text_from_pdf_with_ocr(self, pdf_path: str):
        try:
            import fitz
            doc = fitz.open(pdf_path)
            texts, confs = [], []

            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                results = self.ocr_reader.readtext(img)
                for (_, text, c) in results:
                    if c > 0.3:
                        texts.append(text)
                        confs.append(c)

            return "\n".join(texts), (sum(confs) / len(confs) if confs else 0.0)

        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return "", 0.0

    # -------------------------------------------------------------------------
    # LANGUAGE DETECTION + TRANSLATION (unchanged)
    # -------------------------------------------------------------------------

    def detect_language_with_confidence(self, text: str) -> LanguageDetectionResult:
        if not text.strip():
            return LanguageDetectionResult("en", 0.0, False, True)

        try:
            if self.language_detector["type"] == "advanced":
                try:
                    results = self.language_detector["model"](text[:512])
                    best = max(results[0], key=lambda x: x["score"])
                    lang = best["label"].split("_")[-1]
                    confidence = best["score"]

                    mapping = {
                        "en": "en", "es": "es", "fr": "fr", "de": "de",
                        "zh": "zh", "ja": "ja", "ko": "ko", "ar": "ar",
                        "ru": "ru", "pt": "pt", "it": "it", "nl": "nl",
                    }
                    lang = mapping.get(lang, "en")

                    return LanguageDetectionResult(lang, confidence, confidence > 0.7, False)

                except Exception:
                    pass

            # fallback
            try:
                lang = langdetect.detect(text)
                conf = 0.8
                if lang not in self.supported_languages:
                    lang = "en"
                    conf = 0.5
                return LanguageDetectionResult(lang, conf, conf > 0.6, True)
            except Exception:
                return LanguageDetectionResult("en", 0.3, False, True)

        except Exception:
            return LanguageDetectionResult("en", 0.0, False, True)

    # translation
    def translate_text(self, text: str, src: str, tgt: str = "en"):
        if src == tgt or not text.strip():
            return text, "none"

        key = f"{src}-{tgt}"
        try:
            model = None

            if key in self.translation_models and self.translation_models[key]:
                model = self.translation_models[key]
            elif self.translation_models["main"]:
                model = self.translation_models["main"]

            if not model:
                return text, "no_model_available"

            return (
                self._translate_with_model(text, model["model"], model["tokenizer"]),
                model["name"]
            )
        except Exception:
            return text, "error"

    def _translate_with_model(self, text, model, tokenizer):
        try:
            inp = tokenizer(f">>en<< {text}", return_tensors="pt", truncation=True)
            out = model.generate(**inp)
            return tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception:
            return text

    # -------------------------------------------------------------------------
    # END-TO-END PROCESSING
    # -------------------------------------------------------------------------

    def process_document(self, file_path: str, auto_translate: bool = True):
        start = datetime.now()

        text, extraction = self.extract_text_from_document(file_path)

        if not text.strip():
            return {"success": False, "error": "No text extracted", "file_path": file_path}

        lang = self.detect_language_with_confidence(text)

        translated = text
        model_used = "none"
        if auto_translate and lang.language != "en":
            translated, model_used = self.translate_text(text, lang.language, "en")

        document = Document(
            page_content=translated,
            metadata={
                "source": file_path,
                "original_content": text,
                "original_language": lang.language,
                "detected_language": lang.language,
                "language_confidence": lang.confidence,
                "is_translated": translated != text,
                "translation_model": model_used,
                "processing_metadata": extraction,
            },
        )

        chunks = self.text_splitter.split_documents([document])

        metadata = ProcessingMetadata(
            document_type=DocumentType(extraction["document_type"]),
            original_language=lang.language,
            detected_language=lang.language,
            confidence_score=lang.confidence,
            is_translated=translated != text,
            translation_model=model_used,
            ocr_used=extraction["extraction_method"] == "ocr",
            ocr_confidence=extraction.get("ocr_confidence"),
            processing_time=(datetime.now() - start).total_seconds(),
            document_hash=self._calculate_document_hash(text),
            file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            page_count=len(chunks),
            chunk_count=len(chunks),
        )

        return {
            "success": True,
            "document": document,
            "chunks": chunks,
            "processing_metadata": metadata,
            "language_detection": lang,
            "extraction_metadata": extraction,
        }

    def _calculate_document_hash(self, content: str):
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    # batch processing
    def process_documents_batch(self, paths: List[str], auto_translate: bool = True):
        return [self.process_document(p, auto_translate) for p in paths]

    def get_processing_statistics(self, results: List[Dict[str, Any]]):
        total = len(results)
        success = sum(1 for r in results if r["success"])
        fail = total - success

        lang_counts = {}
        translations = 0
        ocr = 0

        for r in results:
            if r["success"]:
                meta = r["processing_metadata"]
                lang_counts[meta.detected_language] = lang_counts.get(meta.detected_language, 0) + 1
                if meta.is_translated:
                    translations += 1
                if meta.ocr_used:
                    ocr += 1

        return {
            "total_documents": total,
            "successful_documents": success,
            "failed_documents": fail,
            "success_rate": success / total if total > 0 else 0,
            "languages_detected": lang_counts,
            "translations_performed": translations,
            "ocr_documents_processed": ocr,
        }


# Factory
def create_enhanced_processor(enable_ocr: bool = True, supported_languages: List[str] = None,
                              chunk_size: int = 1000, chunk_overlap: int = 200):
    return EnhancedMultilingualProcessor(
        supported_languages=supported_languages,
        enable_ocr=enable_ocr,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )