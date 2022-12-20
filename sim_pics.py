from sklearn.neighbors import NearestNeighbors
import streamlit as st
import pandas as pd
import torch
import open_clip
import PIL
import json

def vectorize(model,img,preprocess):
    with torch.no_grad():
        image = preprocess(img).unsqueeze(0)
        vec = model.encode_image(image).cpu().detach().numpy()[0]
        return vec

@st.cache
def load_base():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
    db = pd.read_csv('base.csv', delimiter=',')
    db['Vec'] = db['Vec'].apply(lambda x: json.loads(x))    
    return model,preprocess,db

model,preprocess, db = load_base()
st.header("Similar Image Search")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "tiff"])
if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.success("This is your image:")
    st.image(image)
    number = st.number_input('Insert a number of neighbors',min_value=1,max_value=10,step=1)
    neighbors_model = NearestNeighbors(n_neighbors=number, metric='cosine')
    neighbors_model.fit(db['Vec'].tolist())
    
    st.success("This is what I found:")
    
    neighbors = neighbors_model.kneighbors([vectorize(model,image,preprocess)], number, return_distance=False)[0]
    col = st.columns(3)
    for i in range(len(neighbors)):
        with col[i%3]:
            st.image(PIL.Image.open(db['Path'][neighbors[i]]))
