from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io
#pdf service
class PDFService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    def extract_and_chunk_pdf(self,file_bytes:bytes)->list[str]:
        pdf_file=io.BytesIO(file_bytes)
        reader= PdfReader(pdf_file)
        raw_text=""
        for page in reader.pages:
            text=""
            text+=page.extract_text()
            if text:
                raw_text+=text+"\n"
        chunks=self.text_splitter.split_text(raw_text)
        return chunks
    
