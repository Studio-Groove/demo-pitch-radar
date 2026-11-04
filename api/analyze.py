from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import librosa

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_response(400, {"error": "Corpo da requisição vazio"})
                return

            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            pcm_data = data.get('audio_data', [])
            sr = int(data.get('sample_rate', 16000))

            if not pcm_data or len(pcm_data) == 0:
                self._send_response(400, {"error": "Dados de áudio vazios"})
                return

            # Conversão segura para o Numpy Array exigido pelo Librosa
            y = np.array(pcm_data, dtype=np.float32)

            # 1. Cálculo do BPM com tratamento seguro
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm_val = float(np.atleast_1d(tempo)[0])
                bpm = int(round(bpm_val)) if not np.isnan(bpm_val) else 124
            except:
                bpm = 124 # Valor padrão seguro de Tech House caso o trecho seja curto

            # 2. Estimativa de Tonalidade (Chroma STFT) segura
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_vals = np.mean(chroma, axis=1)
                notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key_index = int(np.argmax(chroma_vals))
                detected_key = notes[key_index]
            except:
                detected_key = "Am"

            # 3. Estimativa de Loudness Integrado (LUFS) segura
            try:
                rms = librosa.feature.rms(y=y)[0]
                mean_rms = np.mean(rms) + 1e-9
                lufs = round(float(20 * np.log10(mean_rms) - 3.0), 1)
                if np.isnan(lufs) or np.isinf(lufs):
                    lufs = -9.5
            except:
                lufs = -9.5

            payload = {
                "status": "success",
                "audited_bpm": bpm,
                "audited_key": detected_key,
                "audited_lufs": lufs
            }

            self._send_response(200, payload)

        except Exception as e:
            # Retorna o erro detalhado em JSON para sabermos exatamente se houver exceção
            self._send_response(500, {"error": str(e)})

    def _send_response(self, status_code, body):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(body).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
