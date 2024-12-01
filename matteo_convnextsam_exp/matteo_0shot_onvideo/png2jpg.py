import os
from PIL import Image

# Percorsi delle cartelle
cartella_origine = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\images\train"  # Sostituisci con il percorso della tua cartella
cartella_destinazione = r"C:\Users\User\Desktop\datasets\mmi\dataset_video\dataset_video\images\train_jpeg"  # Sostituisci con il percorso della cartella di destinazione

# Crea la cartella di destinazione se non esiste
os.makedirs(cartella_destinazione, exist_ok=True)

# Itera attraverso i file nella cartella di origine
for file in os.listdir(cartella_origine):
    if file.endswith(".png"):
        # Percorso completo dell'immagine di origine
        percorso_origine = os.path.join(cartella_origine, file)

        # Estrai il numero dal nome del file
        numero = file.split("_")[1].split(".")[0]

        # Percorso completo dell'immagine di destinazione
        percorso_destinazione = os.path.join(cartella_destinazione, f"{numero}.jpg")

        # Apri l'immagine, convertila e salvala
        with Image.open(percorso_origine) as img:
            img.convert("RGB").save(percorso_destinazione, "JPEG")

print("Conversione completata!")
