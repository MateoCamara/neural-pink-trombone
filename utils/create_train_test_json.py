import json

# Cargar el archivo JSON original
with open('../../neural-pink-trombone-data/pt_dataset_dynamic_simplified/params.json', 'r') as file:
    data = json.load(file)

sorted_keys = sorted(data.keys())

# Dividir las claves en conjuntos de entrenamiento y prueba
train_keys = sorted_keys[:80000]  # Primeras 80000 claves para entrenamiento
test_keys = sorted_keys[80000:]   # Las siguientes 20000 claves para prueba

# Crear diccionarios para entrenamiento y prueba
train_data = {key: data[key] for key in train_keys}
test_data = {key: data[key] for key in test_keys}

# Guardar los datos de entrenamiento en train.json
with open('../../neural-pink-trombone-data/pt_dataset_dynamic_simplified/train.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)  # Usar indentación para un formato más legible

# Guardar los datos de prueba en test.json
with open('../../neural-pink-trombone-data/pt_dataset_dynamic_simplified/test.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=4)
