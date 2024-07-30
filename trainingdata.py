# -*- coding: utf-8 -*-
"""TrainingData

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e2PoLwR7mLndUi4WdUEsg7m3XG7BWXoM

# Load Data
"""

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

from google.colab import drive
drive.mount('/content/drive')

!pip install unrar

!unrar x "/content/drive/MyDrive/train.rar" "./data"

"""Số lượng floder = class

"""

import os
print("floder-class:", len(os.listdir('./data/train')))

!pip install pytube --upgrade
!pip install fastapi
!pip install kaleido
!pip install python-multipart
!pip install uvicorn
!pip install pyparsing==2.4.7
!pip install numpy==1.23.5

"""#Build clip image embeddings"""

!pip install clip-retrieval autofaiss

!clip-retrieval inference --input_dataset ./data/train/ --output_folder ./data/embedding_folder

!ls ./data/embedding_folder

from pathlib import Path
import pandas as pd
data_dir = Path("./data/embedding_folder/metadata")
df = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)
image_list = df["image_path"].tolist()

image_list

"""# Build a KNN index"""

!autofaiss build_index --embeddings="./data/embedding_folder/img_emb" \
                    --index_path="./data/knn.index" \
                    --index_infos_path="./data/infos.json" \
                    --metric_type="ip" \
                    --max_index_query_time_ms=10 \
                    --max_index_memory_usage="4GB"