# src/logging_config.py
import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Konfiguriert strukturiertes Logging für die gesamte Pipeline.

    - Console: INFO und höher (kompakt)
    - Datei: DEBUG und höher (vollständig, für Nachverfolgbarkeit)
    """
    Path(log_dir).mkdir(exist_ok=True)

    fmt = ("%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(name)s | %(message)s")
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel("DEBUG")

    # Konsole: kompakt
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Datei: vollständig
    file_handler = logging.FileHandler(
        Path(log_dir) / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Drittbibliotheken dämpfen
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)