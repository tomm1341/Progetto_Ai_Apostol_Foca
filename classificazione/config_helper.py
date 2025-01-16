import json
import jsonschema
from pathlib import Path
from types import SimpleNamespace
import sys

def check_and_get_configuration(filename: str, validation_filename: str) -> object:
    json_object = None

    # Indico dove trovare i json di configurazione e di verifica.
    data_file = Path(filename)
    schema_file = Path(validation_filename)

    print(f'Config file is in path: {data_file}')
    print(f'Config file must follow schema rules from path: {schema_file}')

    # Controllo che i file esistano e siano dei json.
    if (data_file.is_file() and schema_file.is_file() and 
        data_file.suffix == '.json' and schema_file.suffix == '.json'):

        with open(data_file) as d:
            with open(schema_file) as s:

                # Carico i due json e utilizzo lo schema per validare il file di configurazione.
                data = json.load(d)
                schema = json.load(s)

                try:
                    jsonschema.validate(instance=data, schema=schema)
                except jsonschema.exceptions.ValidationError:
                    print(f'Json config file is not following schema rules.')
                    sys.exit(-1)
                except jsonschema.exceptions.SchemaError:
                    print(f'Json config schema file is invalid.')
                    sys.exit(-1)

                # Se il file di configurazione è valido, converto il JSON in un oggetto Python.
                json_object = json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))

    if json_object is None:
        print("Errore nel caricamento della configurazione.")
        sys.exit(-1)

    # Se la configurazione è valida, restituisco l'oggetto Python
    return json_object
