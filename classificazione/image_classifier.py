import sys
from net_runner import NetRunner
from config_helper import check_and_get_configuration
from custom_dataset_vehicles import CustomDatasetVehicles
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    cfg_obj = check_and_get_configuration('./classificazione/config/config.json', 
                                       './classificazione/config/config_schema.json')


    # Verifica che la configurazione sia stata caricata correttamente
    if cfg_obj is None:
        print("Errore nel caricamento della configurazione.")
        sys.exit(-1)

    # Usa un data loader per ricavare le classi del dataset.
    classes = CustomDatasetVehicles(root=cfg_obj.io.training_folder, transform=None).classes

    # Crea l'oggetto che permetterà di addestrare e testare il modello.
    runner = NetRunner(cfg_obj, classes)

    # Se richiesto, eseguo il training.
    if cfg_obj.parameters.train:
        runner.train()

    # Se richiesto, eseguo il test.
    if cfg_obj.parameters.test:
        runner.test(print_acc=True)
