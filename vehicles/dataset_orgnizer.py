
import os
import shutil
import random
from pathlib import Path


class DatasetOrganizer:

    def __init__(self, dataset_folder: str, out_folder: str) -> None:
        # Percorsi delle cartelle
        self.dataset_root = Path(dataset_folder)
        self.out_path = Path(out_folder)
        
        # Controlla se il percorso di output esiste e se è una cartella
        if self.out_path.exists():
            if not self.out_path.is_dir():
                print(f'Directory {self.out_path.as_posix()} is not a valid folder')
                exit(-1)
            # Se la cartella contiene già delle immagini, rimuovo tutto
            if not next(self.out_path.iterdir(), None) is None:
                shutil.rmtree(self.out_path)
        
        # Creo la cartella di output principale
        self.out_path.mkdir()

        # Creo le cartelle di destinazione: training, validation, test
        self.tr_path = self.out_path / 'training'
        self.te_path = self.out_path / 'test'
        self.va_path = self.out_path / 'validation'

        # Crea le cartelle di destinazione se non esistono
        for path in [self.tr_path, self.te_path, self.va_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Percentuali di suddivisione (fisse)
        self.train_percentage = 0.6
        self.val_percentage = 0.2
        self.test_percentage = 0.2

        if self.train_percentage + self.val_percentage + self.test_percentage != 1.0:
            print(f"Error: The sum of train, validation, and test percentages must equal 1.")
            exit(-1)

        # Recupero i nomi delle classi (cartelle) nel dataset
        self.classes = sorted([f.name for f in self.dataset_root.iterdir() if f.is_dir()])
        print(f'Classes found: {self.classes}')

        # Suddivido le immagini
        self.split_images()

    def split_images(self):
        """Suddivide le immagini nelle cartelle di destinazione (training, validation, test)."""
        # Inizializzo un dizionario per tenere traccia delle immagini per classe
        image_paths = {}

        # Recupero i percorsi delle immagini per ogni classe
        for class_name in self.classes:
            class_folder = self.dataset_root / class_name
            if class_folder.is_dir():
                images = list(class_folder.glob("*.jpg"))
                print(f"Found {len(images)} images in class '{class_name}'")  # Aggiungi una stampa
                image_paths[class_name] = images

        # Verifica se ci sono immagini in ogni classe
        if not image_paths:
            print("No images found in the dataset.")
            exit(-1)

        # Suddivido le immagini nelle cartelle di training, validation e test
        for class_name, images in image_paths.items():
            # Mescolo le immagini casualmente
            random.shuffle(images)

            # Calcolo la quantità di immagini per ciascun gruppo
            num_test = int(len(images) * self.test_percentage)
            num_val = int(len(images) * self.val_percentage)
            num_train = len(images) - num_test - num_val

            print(f"Class '{class_name}': {len(images)} images")
            print(f"Training: {num_train}, Validation: {num_val}, Test: {num_test}")

            # Assegno le immagini ai gruppi
            train_images = images[:num_train]
            val_images = images[num_train:num_train + num_val]
            test_images = images[num_train + num_val:]

            # Copio le immagini nelle rispettive cartelle
            self.copy_images(train_images, self.tr_path, class_name)
            self.copy_images(val_images, self.va_path, class_name)
            self.copy_images(test_images, self.te_path, class_name)

    def copy_images(self, images, dest_folder, class_name):
        """Copia le immagini nella cartella di destinazione per ciascuna classe."""
        class_folder = dest_folder / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for img in images:
            shutil.copy(img, class_folder / img.name)


if __name__ == '__main__':
    # Percorsi per il dataset e la cartella di output
    dataset_folder = "./vehicles/Vehicles"  
    out_folder = "./out"          

    # Crea un oggetto DatasetOrganizer e avvia la suddivisione
    organizer = DatasetOrganizer(dataset_folder, out_folder)

