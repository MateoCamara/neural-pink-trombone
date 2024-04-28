from pt_data_loader_online import PTServidorDataset

dataset = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1)
dataset.generate_records('../../pt_dataset_simplified', num_records=int(1e5))

dataset = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1, number_of_changes=2)
dataset.generate_records('../../pt_dataset_dynamic_simplified', num_records=int(1e5))