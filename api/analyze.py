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
            
            pcm_data = data.get('audio_data')
            sr = data.get('sample_rate', 16000)

            if not pcm_data:
                self._send_response(400, {"error": "Dados de áudio não fornecidos"})
                return

            y = np.array(pcm_data, dtype=np.float32)

            if len(y) == 0:
                self._send_response(400, {"error": "Buffer de áudio vazio"})
                return

            # 1. Cálculo Real de Tempo / BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = int(round(float(np.atleast_1d(tempo)[0])))

            # 2. Estimativa de Tonalidade via Chroma STFT
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_vals = np.mean(chroma, axis=1)
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_index = int(np.argmax(chroma_vals))
            detected_key = notes[key_index]

            # 3. Estimativa de Loudness Integrado (LUFS)
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms) + 1e-9
            lufs = round(float(20 * np.log10(mean_rms) - 3.0), 1)

            payload = {
                "status": "success",
                "audited_bpm": bpm,
                "audited_key": detected_key,
                "audited_lufs": lufs
            }

            self._send_response(200, payload)

        except Exception as e:
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
