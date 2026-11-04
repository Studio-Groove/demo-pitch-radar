from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io

app = FastAPI(title="Hertz Cartel DSP Engine", version="6.0")

# Habilitar CORS para o frontend falar com o backend localmente
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.aiff')):
        raise HTTPException(status_code=400, detail="Formato de áudio não suportado.")
    
    try:
        # Ler os bytes do arquivo enviado
        contents = await file.read()
        audio_data, samplerate = sf.read(io.BytesIO(contents))
        
        # Se for estéreo, converter para mono para análise espectral balanceada
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data

        # 1. Cálculo de Peak e RMS Real
        peak = float(np.max(np.abs(audio_mono)))
        rms = float(np.sqrt(np.mean(audio_mono**2)))
        crest_factor = float(peak / (rms if rms > 0 else 1e-5))
        
        # 2. Análise Espectral Multibanda (Simulando FFT nas faixas críticas da E-Music)
        # Dividindo o array em blocos para extrair energia aproximada por banda
        n_samples = len(audio_mono)
        chunk = n_samples // 3
        
        sub_band = audio_mono[:chunk]       # Simulação de graves/sub
        punch_band = audio_mono[chunk:2*chunk] # Simulação de punch/transientes
        highs_band = audio_mono[2*chunk:]   # Simulação de agudos/air

        sub_energy = float(np.sqrt(np.mean(sub_band**2)) * 100)
        punch_energy = float(np.sqrt(np.mean(punch_band**2)) * 100)
        highs_energy = float(np.sqrt(np.mean(highs_band**2)) * 100)

        # Normalizar scores para a escala de 0 a 100 do gráfico de radar
        sub_score = min(98, max(50, int(sub_energy * 250)))
        punch_score = min(98, max(50, int(punch_energy * 260)))
        highs_score = min(98, max(50, int(highs_energy * 270)))
        
        # Cálculo de Match de Mercado com base na densidade técnica
        market_match = min(96, max(60, int((sub_score + punch_score + highs_score) / 3)))

        return {
            "filename": file.filename,
            "sample_rate": samplerate,
            "duration": round(n_samples / samplerate, 2),
            "metrics": {
                "peak_db": round(20 * np.log10(peak if peak > 0 else 1e-5), 2),
                "rms": round(rms, 4),
                "crest_factor": round(crest_factor, 2),
                "sub_score": sub_score,
                "punch_score": punch_score,
                "highs_score": highs_score,
                "market_match": market_match
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento DSP: {str(e)}")