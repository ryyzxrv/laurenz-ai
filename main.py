# File: main.py
# (Versi lengkap dengan CORS untuk web)

import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI

# === TAMBAHKAN IMPORT INI ===
from fastapi.middleware.cors import CORSMiddleware
# =============================

# --- 1. Konfigurasi Kunci API ---
# Pastikan Anda mengatur ini di terminal Anda
# (cth: export GEMINI_API_KEY=...)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not GEMINI_API_KEY or not OPENAI_API_KEY:
    print("Peringatan: Pastikan GEMINI_API_KEY dan OPENAI_API_KEY sudah diatur.")

# --- 2. Inisialisasi Klien API ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Peringatan: Gagal inisialisasi klien API. {e}")

app = FastAPI(title="AI Aggregator Konsensus API")

# === 3. Pengaturan CORS (PENTING UNTUK WEB) ===
# Ini mengizinkan browser (dari 'null' / file://) untuk mengakses API Anda
origins = [
    "null", # Untuk mengizinkan permintaan dari file://
    "http://127.0.0.1",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Izinkan semua metode (POST, GET, dll)
    allow_headers=["*"], # Izinkan semua header
)
# =================================================

# --- 4. Definisi Model Data (Pydantic) ---
class PromptRequest(BaseModel):
    prompt: str
    models: List[str] = ["gemini", "gpt"]

class ModelResponse(BaseModel):
    model_name: str
    response: str

# --- 5. Fungsi Helper Pemanggil LLM ---
async def call_llm(model_name: str, prompt: str) -> ModelResponse:
    """Memanggil LLM yang dipilih secara asynchronous."""
    try:
        if model_name == "gemini":
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            return ModelResponse(model_name="Gemini", response=response.text)
        
        elif model_name == "gpt":
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-3.5-turbo", # atau "gpt-4"
                messages=[{"role": "user", "content": prompt}]
            )
            return ModelResponse(model_name="GPT (OpenAI)", response=response.choices[0].message.content)
        
        else:
            return ModelResponse(model_name=model_name, response="Model tidak dikenal.")
            
    except Exception as e:
        print(f"Error memanggil {model_name}: {e}")
        return ModelResponse(model_name=model_name, response=f"Gagal mendapatkan respons: {e}")

# --- 6. Fungsi Logika Konsensus ---
async def generate_consensus(prompt: str, responses: List[ModelResponse]) -> str:
    """Menghasilkan ringkasan konsensus dari berbagai respons LLM."""
    
    # Jika hanya satu respons atau terjadi error, kembalikan respons pertama
    if len(responses) == 1:
        return responses[0].response

    # Buat prompt baru untuk AI konsensus
    consensus_prompt = f"""
    Tugas Anda adalah bertindak sebagai editor ahli.
    Anda telah menerima beberapa jawaban dari AI yang berbeda untuk pertanyaan awal pengguna.
    
    Pertanyaan Pengguna: "{prompt}"

    Berikut adalah jawaban-jawaban tersebut:
    """
    
    for resp in responses:
        consensus_prompt += f"\n--- Jawaban dari {resp.model_name} ---\n{resp.response}\n--- Akhir Jawaban ---\n"
        
    consensus_prompt += """
    Harap sintesiskan jawaban-jawaban ini menjadi satu jawaban akhir yang kohesif, akurat, dan komprehensif. 
    Ambil poin-poin terbaik dari setiap jawaban. Jangan hanya mendaftar apa yang dikatakan setiap AI.
    Tuliskan jawaban akhir seolah-olah Anda menjawab pertanyaan pengguna secara langsung.
    """
    
    try:
        # Menggunakan Gemini untuk membuat konsensus
        consensus_response = await asyncio.to_thread(gemini_model.generate_content, consensus_prompt)
        return consensus_response.text
    except Exception as e:
        print(f"Error saat membuat konsensus: {e}")
        return f"Gagal membuat konsensus. Error: {e}"

# --- 7. API Endpoint Utama ---
@app.post("/api/aggregate", response_model=ModelResponse)
async def aggregate_responses(request: PromptRequest):
    """
    Menerima prompt, mengirimkannya ke beberapa LLM, 
    dan mengembalikan konsensus.
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt tidak boleh kosong.")
        
    # Panggil semua model secara paralel
    tasks = [call_llm(model_name, request.prompt) for model_name in request.models]
    model_responses = await asyncio.gather(*tasks)
    
    # Filter respons yang gagal
    successful_responses = [r for r in model_responses if "Gagal mendapatkan respons" not in r.response]
    
    if not successful_responses:
        raise HTTPException(status_code=500, detail="Semua model AI gagal merespons.")
        
    # Hasilkan konsensus
    consensus_text = await generate_consensus(request.prompt, successful_responses)
    
    return ModelResponse(model_name="Konsensus", response=consensus_text)

# --- 8. (Opsional) Endpoint Root ---
@app.get("/")
def read_root():
    return {"status": "AI Aggregator API sedang berjalan!"}

