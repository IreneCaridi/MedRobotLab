import os
import cv2
import numpy as np

''' codice che serve a estrarre le maschere. prima runnare:
python segment/predict.py --weights '' --source '' --save-txt --nosave     
poi inserire i percorsi di immagini e labels e runnare il codice
'''

def extract_pixels(lista, h, w):
    clas = int(lista[0])
    pix = []
    # Usa flag per alternare e raccogliere le coordinate
    for i in range(1, len(lista) - 1, 2):
        x = int(float(lista[i])*w)
        y = int(float(lista[i + 1])*h)
        pix.append([x, y])  # Aggiungi coppia di coordinate come nuova lista
    return pix, clas

def polytomask(label_folder, image_folder, output_folder):
    '''computes the mask from polygonal label file
    label_folder: folder with label files
    image_folder: folder with image files
    output_folder: folder to save the masks'''
    for file in os.listdir(image_folder):
        name = file.split(".")[0]
        if os.path.exists(os.path.join(label_folder, f'{name}.txt')):
            img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_UNCHANGED)
            height, width = img.shape[:2]

            with open(os.path.join(label_folder, f'{name}.txt')) as txt:
                txt_lines = txt.readlines()

            # Creazione della maschera
            mask = np.zeros(img.shape[:2], np.uint8)

            for line in txt_lines: # per ogni oggetto
                line = line.strip() #tolgo \n
                lista = line.split(' ') #splitto le coordinate
                pixels, classe = extract_pixels(lista, height, width) #estraggo la lista di pixel e la classe
                if len(pixels) != 0:
                    cv2.drawContours(mask, [np.array(pixels, dtype=np.int32)], -1, 50*(classe+1), -1, cv2.LINE_AA) #disegno l'oggetto sulla maschera

            # Verifica se la cartella esiste
            if not os.path.exists(output_folder):
                # Crea la cartella
                os.makedirs(output_folder)

            # Specifica il nome del file di output con estensione
            output_path = os.path.join(output_folder, f"{name}.png")
            cv2.imwrite(output_path, mask)
