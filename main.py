# File: main.py
import os
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from openai import OpenAI

# --- Konfigurasi Kunci API (Disarankan menggunakan file .env) ---
# Dapatkan kunci API dari environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Inisialisasi Klien API
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Peringatan: Gagal inisialisasi klien API. {e}")

app = FastAPI(title="AI Aggregator Konsensus API")

# --- Skema Data Pydantic ---
class PromptRequest(BaseModel):
    prompt: str

class ModelResponse(BaseModel):
    model_name: str
    response: str
    is_successful: bool

class AggregatedResponse(BaseModel):
    final_consensus_answer: str
    individual_responses: List[ModelResponse]

# --- Fungsi Panggilan Asinkronus ---

async def call_llm(client, model_name: str, prompt: str, is_gemini: bool = True) -> ModelResponse:
    """Fungsi pembantu untuk memanggil LLM (Gemini atau OpenAI)"""
    try:
        if is_gemini:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
            )
            text = response.text
        else: # OpenAI
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            
        return ModelResponse(
            model_name=model_name,
            response=text,
            is_successful=True
        )
    except Exception as e:
        return ModelResponse(
            model_name=model_name,
            response=f"Error: Gagal memanggil {model_name}. {e}",
            is_successful=False
        )

# --- Logika Konsensus Kompleks ---

async def generate_consensus(responses: List[ModelResponse], original_prompt: str) -> str:
    """
    Logika Agregasi: Menggunakan Model Juri (Gemini-Pro) untuk
    merangkum dan memilih jawaban terbaik.
    """
    successful_responses = [res for res in responses if res.is_successful]
    
    if not successful_responses:
        return "Maaf, tidak ada model yang berhasil memberikan jawaban."

    # 1. Kumpulkan semua jawaban sukses
    combined_answers = "\n\n".join(
        [f"[{res.model_name}]: {res.response}" for res in successful_responses]
    )

    # 2. Buat prompt untuk Model Juri (menggunakan GPT-4o sebagai contoh Juri)
    jury_prompt = (
        "Anda adalah Juri AI. Tugas Anda adalah membaca prompt asli di bawah, "
        "kemudian menganalisis jawaban dari beberapa model AI yang berbeda, "
        "dan merangkumnya menjadi jawaban tunggal yang paling akurat, komprehensif, dan terbaik."
        "\n\n--- PROMPT ASLI ---\n"
        f"{original_prompt}"
        "\n\n--- JAWABAN MODEL LAIN (Analisis ini) ---\n"
        f"{combined_answers}"
        "\n\n--- JAWABAN KONSENSUS TUNGGAL (Jawaban Anda) ---"
    )

    # 3. Panggil Model Juri (Di sini kita menggunakan GPT-4o karena kompleksitasnya)
    try:
        jury_response = await call_llm(
            client=openai_client, 
            model_name="gpt-4o", # Model paling canggih sebagai Juri
            prompt=jury_prompt,
            is_gemini=False
        )
        if jury_response.is_successful:
            return jury_response.response
        else:
            return f"Logika Konsensus gagal: {jury_response.response}. Mengembalikan semua jawaban individu."
    except Exception:
        # Fallback jika panggilan Juri gagal
        return "Logika Konsensus gagal. Mengembalikan gabungan jawaban mentah."

# --- ENDPOINT UTAMA ---

@app.post("/api/aggregate")
async def aggregate_ai_responses(request: PromptRequest) -> AggregatedResponse:
    prompt = request.prompt

    # 1. Menjalankan semua panggilan API secara paralel (Model yang Saling Bekerja)
    tasks = [
        call_llm(gemini_client, "gemini-2.5-flash", prompt, True), # Pekerja 1
        call_llm(openai_client, "gpt-3.5-turbo", prompt, False),  # Pekerja 2
        # --- Tambahkan Ryhar API di sini jika detailnya diketahui ---
        # call_ryhar_api_async(ryhar_client, prompt) # Pekerja 3
    ]
    
    individual_responses: List[ModelResponse] = await asyncio.gather(*tasks)

    # 2. Logika Konsensus (Model Juri bekerja)
    final_answer = await generate_consensus(individual_responses, prompt)

    return AggregatedResponse(
        final_consensus_answer=final_answer,
        individual_responses=individual_responses
    )

# --- Cara Menjalankan Back-End ---
# Jalankan server: uvicorn main:app --reload
