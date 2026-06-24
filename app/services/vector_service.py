import chromadb
import requests
from rank_bm25 import BM25Okapi

class VectorService:
    def __init__(self):

        # 1. Initialize Vector Database (Chroma)
        self.chroma_client=chromadb.PersistentClient(path="./data/chroma_db")
        self.collection=self.chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )

         # 2. Initialize Lexical Database (BM25)
        self.bm25 = None
        self.corpus_documents = []
        self._sync_bm25_index()

    def _sync_bm25_index(self):

        all_data= self.collection.get()
        if all_data and "documents" in all_data and len(all_data["documents"]) > 0:
                    if all_data and "documents" in all_data:
                        self.corpus_documents=all_data["documents"]
                        tokenized_corpus=[doc.lower().split() for doc in self.corpus_documents]
                        self.bm25=BM25Okapi(tokenized_corpus)
        else:
            self.corpus_documents=[]
            self.bm25=None

    def generate_embedding(self,text:str):
        url="http://localhost:11434/api/embeddings"

        payload={
            "model":"nomic-embed-text-v2-moe",
            "prompt":text
        }

        response=requests.post(url,json=payload)
        response_data=response.json()

        if "error" in response_data:
            raise Exception(f"Ollama Error: {response_data['error']}")
            
        # Standard local /api/embeddings endpoint returns the key 'embedding'
        if "embedding" in response_data:
            return response_data["embedding"]
        else:
            print("--- Unexpected Response Format ---", response_data)
            raise KeyError("The key 'embedding' was not found in the Ollama response.")
    
    def add_chunks_to_db(self,chunks:list,doc_id:str):
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        embeddings = [self.generate_embedding(chunk) for chunk in chunks]

        metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        self._sync_bm25_index()

    def query_hybrid(self,query_text:str,n_results:int =3):
        
        if not self.corpus_documents or self.bm25 is None:
            print("⚠️ Warning: Database is empty. Skipping search.")
            return []
        
        # 1. Embedding Matching using KNN in Chroma
        prefixed_query=f"search_query: {query_text}"
        query_embedding=self.generate_embedding(prefixed_query)

        results=self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(10,len(self.corpus_documents))
        )
        doc_vector = []
        if results and "documents" in results and results["documents"]:
            doc_vector = results["documents"][0]
            metadata = results["metadatas"][0]
            distance = results["distances"][0]
        
        # 2. Lexical Matching using BM25
        tokenized_query=query_text.lower().split()
        bm25_scores=self.bm25.get_scores(tokenized_query)

        bm25_ranked_indices=sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_docs = [self.corpus_documents[i] for i in bm25_ranked_indices[:10]]

        # 3. Combine Results from Chroma and BM25

        k = 60 # Standard smoothing constant
        rrf_scores = {}

        # Apply RRF math to Vector ranks
        for rank, doc in enumerate(doc_vector):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1 / (k + rank + 1)

        # Apply RRF math to BM25 ranks
        for rank, doc in enumerate(bm25_docs):
            if doc not in rrf_scores:
                rrf_scores[doc] = 0.0
            rrf_scores[doc] += 1 / (k + rank + 1)

        # Sort the fused dictionary by highest RRF score
        sorted_fused_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Return only the text chunks of the absolute top_k winners
        return [doc[0] for doc in sorted_fused_docs[:10]]
    def peek_inside_db(self):
        print("\n--- 🕵️ DATABASE METADATA INSPECTION ---")
        all_data = self.collection.get()
        
        if not all_data or not all_data["metadatas"]:
            print("Database is completely empty.")
            return

        # Print the unique Document IDs currently sitting in the database
        unique_docs = set(meta["doc_id"] for meta in all_data["metadatas"])
        print(f"Total Unique PDFs stored: {len(unique_docs)}")
        for doc in unique_docs:
            print(f" 📄 Found Document ID: {doc}")
        print("----------------------------------------\n")