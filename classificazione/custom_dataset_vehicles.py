import sys
from PIL import Image  
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset



class CustomDatasetVehicles(Dataset):
    def __init__(self, root: str, transform=None, debug: bool = False, resize_dim=(224, 224)) -> None:
        self.debug = debug
        self.transform = transform
        self.data_path = Path(root)
        self.resize_dim = resize_dim  

        # Controlla il percorso
        if not self.__analyze_root():
            sys.exit(-1)

        # Cerca file immagine
        if not self.__search_image_files():
            sys.exit(-1)

        # Controlla la struttura delle cartelle
        if not self.__check_structure():
            sys.exit(-1)

        # Trova le classi e le etichette
        self.__find_classes_and_labels()

        # Assegna le etichette
        self.__assign_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Leggi l'immagine
        img = cv2.imread(self.image_files[index].as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Converte il numpy array in un'immagine PIL
        img_pil = Image.fromarray(img)

        # Normalizza l'immagine (non cambia l'oggetto PIL)
        img_array = np.array(img_pil) / 255.0  # Porta i pixel nel range [0, 1]

        # Ridimensiona l'immagine
        if self.resize_dim:
            img_pil = img_pil.resize(self.resize_dim)

        # Aggiungi rumore gaussiano
        noise = np.random.normal(0, 0.05, img_pil.size + (3,))  # Rumore con deviazione standard 0.05
        img_array = np.array(img_pil) + noise
        img_array = np.clip(img_array, 0, 1)  # Limita il range a [0, 1]

        class_idx = self.labels[index]

        if self.transform:
            return self.transform(img_pil), class_idx
        else:
            return np.array(img_pil).astype(np.float32), class_idx


    def __analyze_root(self) -> bool:
        if self.debug:
            print(f'Analisi del percorso: {self.data_path.as_posix()}')

        if self.data_path.exists():
            if not self.data_path.is_dir():
                if self.debug:
                    print(f'{self.data_path.as_posix()} non e\' una cartella valida.')
                return False
        else:
            if self.debug:
                print(f'Cartella {self.data_path.as_posix()} inesistente.')
            return False

        if self.debug:
            print(f'Il percorso e\' valido.')
        return True

    def __search_image_files(self) -> bool:
        image_extensions = ('.bmp', '.png', '.jpeg', '.jpg')

        self.image_files = [x for x in self.data_path.glob('**/*')
                            if x.is_file() and x.suffix in image_extensions]

        if len(self.image_files) > 0:
            if self.debug:
                print(f'Nella cartella sono stati trovati {len(self.image_files)} file immagine.')
            return True
        else:
            if self.debug:
                print(f'Nessuna immagine valida trovata.')
            return False

    def __check_structure(self) -> bool:
        condition_1 = all(len(f.parts) > 2 for f in self.image_files)
        condition_2 = all(f.parent.parent == self.data_path for f in self.image_files)

        if condition_1 and condition_2:
            if self.debug:
                print(f'La struttura delle sotto-cartelle in {self.data_path} e\' valida.')
            return True
        else:
            if self.debug:
                print(f'La struttura delle sotto-cartelle in {self.data_path} non e\' valida.')
            return False

    def __find_classes_and_labels(self) -> None:
        sub_folder_names = [f.parts[-2] for f in self.image_files]
        self.classes = sorted(list(set(sub_folder_names)))

        counter = 0
        self.class_labels = {}
        for c in self.classes:
            self.class_labels[c] = counter
            counter += 1

        if self.debug:
            print('Classi trovate ed etichette assegnate:')
        for c, l in self.class_labels.items():
            if self.debug:
                print(f'|__Classe [{c}]\t: etichetta [{l}]')

    def __assign_labels(self) -> None:
        self.labels = []
        class_distributions = {c: 0 for c in self.classes}

        for i in self.image_files:
            image_class = i.parts[-2]
            class_distributions[image_class] += 1
            self.labels.append(self.class_labels[image_class])

        if self.debug:
            print('Distribuzione classi:')
        for c, d in class_distributions.items():
            if self.debug:
                print(f'|__Classe [{c}]\t: {d} ({d/float(len(self.image_files)):.2f}%)')


if __name__ == '__main__':
    cds = CustomDatasetVehicles('./out/training', debug=True)

    for i, data in enumerate(cds):
        if i == 30:
            break
        print(f'Campione {i}, etichetta: [{data[1]}]')

        # Mostra l'immagine normalizzata con OpenCV
        cv2.imshow(f"Immagine {i}", (data[0] * 255).astype(np.uint8))  # Denormalizza per la visualizzazione

        # Aspetta un tasto per continuare
        cv2.waitKey(0)

    cv2.destroyAllWindows()  # Chiude tutte le finestre di OpenCV
