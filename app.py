import streamlit as st
from torchvision import models, transforms
import torch
from PIL import Image


def predict(image_path):
    resnet = models.resnet101(pretrained=True)

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Image Classification App")
st.write("")

file_up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.subheader("Prediction Scores")
    st.write("Just a second...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        # st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
        st.success(f"Prediction -> {i[0]},   Score: {i[1]}")