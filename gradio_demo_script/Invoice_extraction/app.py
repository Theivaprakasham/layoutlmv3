import os
os.system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu')

import gradio as gr
import numpy as np
from transformers import AutoModelForTokenClassification
from datasets.features import ClassLabel
from transformers import AutoProcessor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
from datasets import load_metric
from transformers import LayoutLMv3ForTokenClassification
from transformers.data.data_collator import default_data_collator


from transformers import AutoModelForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = AutoModelForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice")



# load image example
dataset = load_dataset("darentang/generated", split="test")
Image.open(dataset[2]["image_path"]).convert("RGB").save("example1.png")
Image.open(dataset[1]["image_path"]).convert("RGB").save("example2.png")
Image.open(dataset[0]["image_path"]).convert("RGB").save("example3.png")
# define id2label, label2color
labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2color = {
    "B-ABN": 'blue',
    "B-BILLER": 'blue',
    "B-BILLER_ADDRESS": 'green',
    "B-BILLER_POST_CODE": 'orange',
    "B-DUE_DATE": "blue",
    "B-GST": 'green',
    "B-INVOICE_DATE": 'violet',
    "B-INVOICE_NUMBER": 'orange',
    "B-SUBTOTAL": 'green',
    "B-TOTAL": 'blue',
    "I-BILLER_ADDRESS": 'blue',
    "O": 'orange'
  } 

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]


def iob_to_label(label):
    return label



def process_image(image):

    print(type(image))
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction)
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    return image


title = "Invoice Information extraction using LayoutLMv3 model"
description = "Invoice Information Extraction - We use Microsoft's LayoutLMv3 trained on Invoice Dataset to predict the Biller Name, Biller Address, Biller post_code, Due_date, GST, Invoice_date, Invoice_number, Subtotal and Total. To use it, simply upload an image or use the example image below. Results will show up in a few seconds."

article="<b>References</b><br>[1] Y. Xu et al., “LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.” 2022. <a href='https://arxiv.org/abs/2204.08387'>Paper Link</a><br>[2]  <a href='https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3'>LayoutLMv3 training and inference</a>" 

examples =[['example1.png'],['example2.png'],['example3.png']]

css = """.output_image, .input_image {height: 600px !important}"""

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image(type="pil", label="annotated image"),
                     title=title,
                     description=description,
                     article=article,
                     examples=examples,
                     css=css,
                     analytics_enabled = True, enable_queue=True)

iface.launch(inline=False, share=False, debug=False)