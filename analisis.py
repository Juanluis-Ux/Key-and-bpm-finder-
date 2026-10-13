import librosa
import numpy as np

y, sr = librosa.load("/Users/paulino/proyectos/3am.wav")
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
brillo = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

mayor = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
menor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
notas = ["Do","Do#","Re","Re#","Mi","Fa","Fa#","Sol","Sol#","La","La#","Si"]

mejor = None
for i in range(12):
    for nombre, perfil in (("mayor", mayor), ("menor", menor)):
        c = np.corrcoef(chroma, np.roll(perfil, i))[0, 1]
        if mejor is None or c > mejor[0]:
            mejor = (c, notas[i], nombre)

print("Tempo:", round(float(np.atleast_1d(tempo)[0])), "BPM")
print("Tonalidad:", mejor[1], mejor[2])
print("Frecuencia:", round(brillo), "Hz")

duracion = len(y) / sr

minutos = int(duracion // 60)
segundos = round(duracion % 60)
print("Duración:", minutos, "min", segundos, "s")   