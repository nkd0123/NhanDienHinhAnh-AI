import faiss # tim kiem do tuong dong hieu qua
import torch # PyTorch tensors - Ho Tro tinh toan nhanh hon tren GPU
import clip 
import os
import pandas as pd

# Read file KNN index - doc file KNN
df = pd.read_parquet(".\data\embedding_folder\metadata\metadata_0.parquet")
image_list = df["image_path"].tolist() # chua duong dan hinh anh
ind = faiss.read_index(".\data\knn.index") # chi muc  knn anh luu tru database

# Load the model - load cau hinh can thiet
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 

# Search image - tim kiem hinh anh
def search_face(image):
    image_tensor = preprocess(image)
    image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_embeddings = image_features.cpu().detach().numpy().astype('float32')
    D, I = ind.search(image_embeddings, 1) 
    if D[0][0] > 0.8: #  do tuong dong >0.8
        name = os.path.basename(os.path.dirname(image_list[I[0][0]])) 
        print("Name:",os.path.basename(os.path.dirname(image_list[I[0][0]])))  # in ra ten
        print("Similarity:",D[0][0]) # do tuong dong
        return name
    

    # D do tuong dong
    # I là đường dẫn của ảnh đó trong database