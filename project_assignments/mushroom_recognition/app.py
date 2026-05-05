import gradio as gr
import keras
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

model = keras.models.load_model(
    "effnet_finetune_best.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

class_names = ["Agaricus", "Amanita", "Boletus", "Cortinarius",
               "Entoloma", "Hygrocybe", "Lactarius", "Russula", "Suillus"]

toxicity_profiles = {
    "Amanita":     [0.90, 0.10, 0.00],
    "Cortinarius": [0.80, 0.20, 0.00],
    "Entoloma":    [0.70, 0.30, 0.00],
    "Agaricus":    [0.20, 0.60, 0.20],
    "Boletus":     [0.10, 0.40, 0.50],
    "Lactarius":   [0.10, 0.50, 0.40],
    "Russula":     [0.10, 0.50, 0.40],
    "Hygrocybe":   [0.00, 0.20, 0.80],
    "Suillus":     [0.00, 0.20, 0.80],
}

finnish_names = {
    "Agaricus": "Herkkusieni",
    "Amanita": "Kärpässieni",
    "Boletus": "Tatti",
    "Cortinarius": "Seitikki",
    "Entoloma": "Malikka",
    "Hygrocybe": "Vahakas",
    "Lactarius": "Rousku",
    "Russula": "Hapero",
    "Suillus": "Tahmatatti"
}

toxicity_weights = np.array([1.0, 0.5, 0.0])

def predict(image):
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    probs = model.predict(img, verbose=0)[0]

    toxicity_score = sum(
        probs[i] * np.dot(toxicity_profiles[class_names[i]], toxicity_weights)
        for i in range(len(class_names))
    )

    risk = "⚠️ High risk" if toxicity_score > 0.6 else "⚡ Medium risk" if toxicity_score > 0.3 else "✅ Low risk"

    return (
        {
            f"{class_names[i]} | {finnish_names[class_names[i]]}": float(probs[i])
            for i in range(len(class_names))
        },
        f"{toxicity_score:.3f} — {risk}"
    )

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=5), gr.Textbox(label="Toxicity Score")],
    title="Mushroom Toxicity Classifier",
    description="Upload a mushroom image to classify its genus and estimate toxicity risk. Please note that this is a student " \
    "project and should not be used for real-life mushroom identification or toxicity assessment."
).launch()