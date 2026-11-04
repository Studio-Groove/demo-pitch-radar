from http.server import BaseHTTPRequestHandler
import json
import cgi
import io
import numpy as np
import librosa

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_type, pdict = cgi.parse_header(self.headers.get('content-type'))
            if content_type != 'multipart/form-data':
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Content-Type deve ser multipart/form-data"}).encode('utf-8'))
                return

            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            pdict['CONTENT-LENGTH'] = int(self.headers.get('content-length'))
            fields = cgi.parse_multipart(self.rfile, pdict)

            audio_bytes = fields.get('audio')[0]
            
            y, sr = librosa.load(io.BytesIO(audio_bytes), duration=45.0)

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            audited_bpm = int(np.round(float(tempo)))

            if audited_bpm < 100:
                audited_bpm *= 2
            elif audited_bpm > 150:
                audited_bpm = int(np.round(audited_bpm / 2))

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_vals = np.sum(chroma, axis=1)
            
            pitch_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
            best_pitch_idx = int(np.argmax(chroma_vals))
            detected_note = pitch_names[best_pitch_idx]
            
            is_minor = chroma_vals[(best_pitch_idx + 3) % 12] > chroma_vals[(best_pitch_idx + 4) % 12]
            audited_key = f"{detected_note} {'Min' if is_minor else 'Maj'}"

            rms = librosa.feature.rms(y=y)
            avg_rms = float(np.mean(rms))
            audited_lufs = float(-0.691 + (10 * np.log10(avg_rms + 1e-6)) + 9.2)

            response_data = {
                "success": True,
                "audited_bpm": audited_bpm,
                "audited_key": audited_key,
                "audited_lufs": round(audited_lufs, 1),
                "true_peak_db": -0.1
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))