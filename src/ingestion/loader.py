import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ReportFile:
    path: Path
    raw_text: str
    filename: str
    parent_folder: str
    original_case_id: Optional[str] = None
    report_year: Optional[int] = None
    encoding: str = "utf-8"


class ReportFileLoader:
	"""Findet .txt-Berichte und liest sie robust ein (Encoding-Fallback).

	Sortierung erfolgt deterministisch nach Dateiname, damit
	"die ersten 100" reproduzierbar sind."""

	def __init__(self, directory: str, pattern: str = "*.txt",
					recursive: bool = False, limit: Optional[int] = None,
					encodings: Optional[list[str]] = None,
					filename_pattern: Optional[str] = None):
		self.directory = Path(directory)
		self.pattern = pattern
		self.recursive = recursive
		self.limit = limit
		self.encodings = encodings or ["utf-8", "cp1252", "latin-1"]
		self.filename_re = re.compile(filename_pattern) if filename_pattern else None

	def discover(self) -> list[Path]:
		if not self.directory.exists():
			raise FileNotFoundError(
				f"Import-Verzeichnis nicht gefunden: {self.directory}")
		files = self.directory.rglob(self.pattern) if self.recursive \
			else self.directory.glob(self.pattern)
		files = sorted(p for p in files if p.is_file())
		return files[: self.limit] if self.limit else files

	def read_file(self, path: Path) -> ReportFile:
		last_exc: Exception | None = None
		for enc in self.encodings:
			try:
				text = path.read_text(encoding=enc)
				return self._build(path, text, enc)
			except UnicodeDecodeError as e:
				last_exc = e
		raise ValueError(f"Kein Encoding aus {self.encodings} passt für {path}") \
			from last_exc

	def extract_year(self, parent_folder: str, filename: str) -> Optional[int]:
		for source in (parent_folder, filename):
			m = re.search(r"(19|20)\d{2}", source or "")
			if m:
				return int(m.group(0))
		return None
		return None

	def _build(self, path: Path, text: str, encoding: str) -> ReportFile:
		# Typische Export-Artefakte bereinigen
		text = text.replace("\r\n", "\n").replace("\x00", "").strip()
		report_year=self.extract_year(path.parent.name, path.name),

		rf = ReportFile(
			path=path, raw_text=text, filename=path.name,
			parent_folder=path.parent.name, encoding=encoding,
			report_year=report_year
		)
		if self.filename_re:
			m = self.filename_re.search(path.stem)
			if m:
				gd = m.groupdict()
				rf.original_case_id = gd.get("case_id")
				if gd.get("year"):
					rf.report_year = int(gd["year"])
		if not rf.original_case_id:
			rf.original_case_id = path.stem   # Fallback: Dateiname ohne Endung
		return rf

	def load(self) -> list[ReportFile]:
		reports: list[ReportFile] = []
		for path in self.discover():
			try:
				rf = self.read_file(path)
			except ValueError as e:
				print(f"WARN: {e} – übersprungen")
				continue
			if not rf.raw_text:
				print(f"WARN: {path.name} ist leer – übersprungen")
				continue
			reports.append(rf)
		return reports