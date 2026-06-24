from fastapi import FastAPI , UploadFile,File,HTTPException
from app.services import PDFService, VectorService, LLMService
import uuid
from pydantic import BaseModel

app = FastAPI(title="PDF Vectorization API")
pdf_service=PDFService()
vector_service=VectorService()
llm_service=LLMService()

class QueryRequest(BaseModel):
    question: str

@app.post("/upload_pdf/")
async def upload_pdf(file:UploadFile=File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
    try:
        file_content_binary=await file.read()
        chunks=pdf_service.extract_and_chunk_pdf(file_content_binary)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found in the PDF.")
        doc_id=str(uuid.uuid4())
        vector_service.add_chunks_to_db(chunks,doc_id)
        vector_service.peek_inside_db()
        return {
            "status": "success",
            "message": f"PDF processed and stored with doc_id: {doc_id}",
            "filename": file.filename,
            "num_chunks": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/query_pdf/")
async def query_pdf(payload: QueryRequest):
    try:
        chunks= vector_service.query_hybrid(payload.question, n_results=3)
        if not chunks:
            return {"answer": "No relevant documents found in the database. Please upload a PDF first."}
        answer=""
        answer="" + llm_service.generate_response(payload.question, chunks)
        return{
            "question": payload.question,
            "answer": answer,   
            "Sources_Referred": len(chunks)
        }
    except Exception as e:
        print(f"Error querying PDF: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")