import torch
import open_clip
from pathlib import Path
import pandas as pd
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')

def vectorize(img):
    with torch.no_grad():
        image = preprocess(img).unsqueeze(0)
        vec = model.encode_image(image).cpu().detach().numpy()[0]
        return vec.tolist()

if not(Path('base.csv').exists()):
    paths = []
    vecs = []
    for file in Path('base').glob('*'):
        paths.append(file)
        vecs.append(vectorize(Image.open(str(file))))
    df = pd.DataFrame(data={'Path':paths, 'Vec':vecs})
    df = df.reset_index()
    df = df.rename(columns={'index':'Id'})
    df.to_csv('base.csv',index=False)
