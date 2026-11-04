from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import librosa
import traceback
import sys

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
            subgenre = data.get('subgenre', 'Tech House')

            if not pcm_data or len(pcm_data) == 0:
                self._send_response(400, {"error": "Dados de áudio vazios"})
                return

            y = np.array(pcm_data, dtype=np.float32)

            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # 1. Extração segura de BPM
            bpm = 128
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm_val = float(np.atleast_1d(tempo)[0])
                if 50 < bpm_val < 90:
                    bpm_val *= 2
                elif bpm_val > 160:
                    bpm_val /= 2
                if not np.isnan(bpm_val) and 90 <= bpm_val <= 150:
                    bpm = int(round(bpm_val))
            except Exception as e:
                print(f"Aviso no cálculo de BPM: {e}", file=sys.stderr)

            # 2. Extração segura de Tonalidade (Chroma STFT)
            detected_key = "Am"
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key_index = int(np.argmax(chroma_mean))
                detected_key = notes[key_index] + "m"
            except Exception as e:
                print(f"Aviso no cálculo de Tonalidade: {e}", file=sys.stderr)

            # 3. Cálculo seguro de Loudness (LUFS)
            lufs = -9.0
            try:
                rms = librosa.feature.rms(y=y)[0]
                mean_rms = np.mean(rms) + 1e-9
                lufs_val = float(20 * np.log10(mean_rms) - 2.5)
                if not np.isnan(lufs_val) and not np.isinf(lufs_val):
                    lufs = round(lufs_val, 1)
            except Exception as e:
                print(f"Aviso no cálculo de LUFS: {e}", file=sys.stderr)

            # 4. Motor de Matching com Gravadoras Reais
            labels_db = {
                "Tech House": [
                    {"name": "Vitrine Records BR", "target_bpm": 128, "target_lufs": -9.0, "fit": 98},
                    {"name": "Solid Grooves Records", "target_bpm": 128, "target_lufs": -8.5, "fit": 95},
                    {"name": "Hot Creations", "target_bpm": 126, "target_lufs": -9.0, "fit": 90},
                    {"name": "Repopulate Mars", "target_bpm": 127, "target_lufs": -8.8, "fit": 88}
                ],
                "Melodic Techno": [
                    {"name": "Afterlife", "target_bpm": 122, "target_lufs": -10.0, "fit": 95},
                    {"name": "Drumcode", "target_bpm": 130, "target_lufs": -8.2, "fit": 90}
                ],
                "Deep House": [
                    {"name": "Defected Records", "target_bpm": 124, "target_lufs": -9.0, "fit": 92},
                    {"name": "Anjunadeep", "target_bpm": 120, "target_lufs": -10.5, "fit": 90}
                ],
                "Techno Peak Time": [
                    {"name": "Drumcode", "target_bpm": 132, "target_lufs": -8.0, "fit": 95},
                    {"name": "Filth on Acid", "target_bpm": 134, "target_lufs": -8.2, "fit": 91}
                ]
            }

            pool = labels_db.get(subgenre, labels_db["Tech House"])
            matched_labels = []
            for item in pool:
                bpm_diff = abs(item["target_bpm"] - bpm)
                lufs_diff = abs(item["target_lufs"] - lufs)
                dynamic_fit = max(50, item["fit"] - int(bpm_diff * 4 + lufs_diff * 3))
                matched_labels.append({
                    "label": item["name"],
                    "match_score": dynamic_fit
                })

            matched_labels = sorted(matched_labels, key=lambda x: x["match_score"], reverse=True)

            payload = {
                "status": "success",
                "audited_bpm": bpm,
                "audited_key": detected_key,
                "audited_lufs": lufs,
                "label_matches": matched_labels
            }

            self._send_response(200, payload)

        except Exception as e:
            # Imprime o erro detalhado nos logs da Vercel se ocorrer exceção crítica
            traceback.print_exc(file=sys.stderr)
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
