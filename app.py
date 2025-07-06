import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
import tempfile
import uuid
import csv
import datetime
import qrcode

st.set_page_config(layout="centered")

model = tf.keras.models.load_model("brain_tumor_cnn_model.h5")
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

st.title("🧠 Brain Tumor MRI Classification with Localization")

st.sidebar.title("📘 About")
st.sidebar.markdown("""
- Uses a trained CNN model
- Predicts from MRI images
- Localizes tumor with Grad-CAM
""")

with open("metrics.txt") as f:
    st.sidebar.info(f.read())

# Patient info
st.sidebar.subheader("🧑 Patient Info")
patient_name = st.sidebar.text_input("Name")
patient_age = st.sidebar.text_input("Age")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d_1', pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img.resize((150, 150)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cm.jet(heatmap)[..., :3] * 255
    superimposed_img = heatmap_colored * alpha + img
    return Image.fromarray(np.uint8(superimposed_img))

uploaded_files = st.file_uploader("Upload MRI image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        st.image(image, caption=file.name, width=250)

        img = image.resize((150, 150))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = model.predict(img_array)[0]
        class_index = np.argmax(prediction)
        confidence = prediction[class_index]

        st.success(f"Prediction: **{classes[class_index]}** with **{confidence*100:.2f}%** confidence.")

        st.subheader("📊 Prediction Probabilities")
        st.bar_chart({classes[i]: prediction[i] for i in range(len(classes))})

        st.subheader("🧼 Preprocessed Image")
        st.image(img, caption="Resized 150x150", width=150)

        st.subheader("🔥 Grad-CAM Heatmap")
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='conv2d_1')
        cam_image = save_and_display_gradcam(img, heatmap)
        st.image(cam_image, caption="Approximate Tumor Area", use_column_width=True)

        result = f"Prediction: {classes[class_index]}\nConfidence: {confidence*100:.2f}%"

        if st.button(f"📄 Generate Report for {file.name}"):
            with tempfile.TemporaryDirectory() as tmpdir:
                original_path = os.path.join(tmpdir, f"{uuid.uuid4()}_input.jpg")
                heatmap_path = os.path.join(tmpdir, f"{uuid.uuid4()}_heatmap.jpg")
                image.save(original_path)
                cam_image.save(heatmap_path)

                # Generate report ID and QR code
                report_id = f"REP-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
                qr_img = qrcode.make(report_id)
                qr_path = os.path.join(tmpdir, "qr_code.png")
                qr_img.save(qr_path)

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.cell(200, 10, txt="Brain Tumor MRI Report", ln=True, align="C")
                pdf.ln(10)
                pdf.set_fill_color(230, 230, 250)
                pdf.cell(0, 10, f"Report ID: {report_id}", ln=True, fill=True)
                pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
                pdf.cell(0, 10, f"Patient Age: {patient_age}", ln=True)
                pdf.cell(0, 10, f"Prediction: {classes[class_index]}", ln=True)
                pdf.cell(0, 10, f"Confidence: {confidence*100:.2f}%", ln=True)

                pdf.ln(5)

                pdf.cell(200, 10, txt="Original Image:", ln=True)
                pdf.image(original_path, x=10, y=None, w=100)

                pdf.cell(200, 10, txt="Grad-CAM Heatmap:", ln=True)
                pdf.image(heatmap_path, x=10, y=None, w=100)

                pdf.ln(10)
                pdf.cell(200, 10, txt="Doctor Signature: ________________________", ln=True)
                pdf.image(qr_path, x=160, y=260, w=30)

                output_pdf = os.path.join(tmpdir, f"{file.name}_report.pdf")
                pdf.output(output_pdf)

                with open(output_pdf, "rb") as pdf_file:
                    st.download_button("⬇️ Download PDF Report", pdf_file.read(), file_name=f"{file.name}_report.pdf")

                # CSV logging
                with open("patient_predictions_log.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if csvfile.tell() == 0:
                        writer.writerow(["Report ID", "Name", "Age", "Image", "Prediction", "Confidence", "Date"])
                    writer.writerow([report_id, patient_name, patient_age, file.name, classes[class_index], f"{confidence*100:.2f}%", datetime.datetime.now().isoformat()])

