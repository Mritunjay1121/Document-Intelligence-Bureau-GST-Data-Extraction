import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class Config:
    """app configuration"""

    # API stuff
    API_TITLE = "Document Intelligence: Bureau & GST Data Extraction"
    API_VERSION = "1.0.0"
    API_HOST = "127.0.0.1"
    API_PORT = 8000

    # OpenAI config
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # model settings
    OPENAI_MODEL = "gpt-4o-mini"  # for RAG
    OPENAI_VISION_MODEL = "gpt-4o"  # for vision extraction
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    OPENAI_TEMPERATURE = 0.0  # deterministic extraction

    # vision settings
    USE_VISION = True  # enable GPT-4 Vision
    VISION_DPI = 200  # image quality for PDF conversion
    VISION_PRIORITY = True  # use vision as primary method

    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_CHUNKS = 5
    SIMILARITY_THRESHOLD = 0.3

    # directories
    BASE_DIR = Path(__file__).parent
    STORAGE_DIR = BASE_DIR / "storage"
    UPLOADS_DIR = STORAGE_DIR / "uploads"
    LOGS_DIR = BASE_DIR / "logs"

    # default file locations
    DEFAULT_BUREAU_DIR = STORAGE_DIR / "bureau_reports"
    DEFAULT_GST_DIR = STORAGE_DIR / "gst_returns"
    DEFAULT_PARAMETERS_DIR = STORAGE_DIR / "parameters"
    DEFAULT_PARAMETERS_EXCEL = DEFAULT_PARAMETERS_DIR / "Bureau-parameters-Report.xlsx"

    @classmethod
    def ensure_directories(cls):
        """create necessary directories"""
        directories = [
            cls.STORAGE_DIR,
            cls.UPLOADS_DIR,
            cls.LOGS_DIR,
            cls.DEFAULT_BUREAU_DIR,
            cls.DEFAULT_GST_DIR,
            cls.DEFAULT_PARAMETERS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_configuration(cls):
        """validate config and check for issues"""
        issues = []

        # check API key
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not set")

        # make sure directories exist
        cls.ensure_directories()

        # check vision settings
        if cls.USE_VISION and not cls.OPENAI_VISION_MODEL:
            issues.append("USE_VISION=True but OPENAI_VISION_MODEL not set")

        # log any issues
        for issue in issues:
            logger.error(f"Config issue: {issue}")

        if len(issues) == 0:
            return True
        return False

    @classmethod
    def get_default_parameters_excel(cls):
        """get default parameters Excel file path"""
        if cls.DEFAULT_PARAMETERS_EXCEL.exists():
            return cls.DEFAULT_PARAMETERS_EXCEL
        return None
