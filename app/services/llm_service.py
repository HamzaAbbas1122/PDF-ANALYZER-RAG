import requests

class LLMService:
    def __init__(self):
        self.base_url = "http://localhost:11434/api/generate"
        self.model_name = "gemma4:31b-cloud"

    def generate_response(self,question:str,contexts: str):
        context_text = "\n---\n".join(contexts)

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context blocks below. "
            "If the answer cannot be found in the context, say 'I cannot find the answer in the provided document.' "
            f"Context:\n{context_text}"
        )

        payload = {
            "model": self.model_name,
            "prompt": question,
            "system": system_prompt,
            "stream": False
        }
        response = requests.post(self.base_url,json=payload)
        response_data = response.json()

        if "error" in response_data:
            raise Exception(f"Ollama Cloud Error: {response_data['error']}")  
        
        return response_data["response"]
    