from pt_data_loader_online import PTServidorDataset

dataset = PTServidorDataset(servidor_url='127.0.0.1', servidor_port=3000, tamano_batch=1, iteraciones=1)
dataset.generate_records('/path/to/output/directory', num_records=1e6)
