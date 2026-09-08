from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.security import validate_filename, validate_upload
from app.ingestion.document_loader import load_document
from app.ingestion.excel_loader import load_excel


class TestSecurity:
    def test_validate_filename_strips_path_traversal(self):
        assert validate_filename("../../etc/passwd") == "passwd"
        assert validate_filename("C:\\Windows\\evil.exe") == "evil.exe"

    def test_validate_upload_rejects_unsupported_extension(self):
        with pytest.raises(UnsupportedFileTypeError):
            validate_upload("malware.exe", 1000)

    def test_validate_upload_rejects_oversized_file(self):
        with pytest.raises(FileTooLargeError):
            validate_upload("doc.pdf", 999_999_999)

    def test_validate_upload_accepts_known_extension(self):
        ext = validate_upload("scheme.pdf", 1000)
        assert ext == ".pdf"


class TestExcelIngestion:
    def test_load_excel_creates_document_per_sheet(self, tmp_path):
        path = tmp_path / "schemes.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"Scheme": ["PM-KISAN"], "Benefit": ["Rs 6000/year"]}).to_excel(writer, sheet_name="Schemes", index=False)
        docs = load_excel(path, source_type="user_upload", title="schemes")
        assert len(docs) == 1
        assert docs[0].extra_metadata["sheet_name"] == "Schemes"
        assert "PM-KISAN" in docs[0].content
        assert docs[0].source_type == "user_upload"

    def test_load_excel_via_dispatcher(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("Scheme,Benefit\nPM-KISAN,Rs 6000/year\n", encoding="utf-8")
        docs = load_document(path, source_type="user_upload")
        assert len(docs) == 1
        assert docs[0].file_type == "excel"
        assert docs[0].category  # metadata_extractor should have set something


class TestPDFIngestion:
    def test_load_pdf_extracts_text(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        path = tmp_path / "notice.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "PM-KISAN provides Rs 6000 per year to eligible farmers.")
        doc.save(path)
        doc.close()

        docs = load_document(path, source_type="user_upload", title="notice")
        assert len(docs) == 1
        assert "PM-KISAN" in docs[0].content
        assert docs[0].file_type == "pdf"
        assert docs[0].source_type == "user_upload"
