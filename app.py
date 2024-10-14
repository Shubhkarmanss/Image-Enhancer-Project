from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import torch
import streamlit as st

app = Flask(__name__)

# Load the ESRGAN model
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
# model.eval()

# Define the image editor function
def merge_images(img1, img2):
    """Merge two images horizontally"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        # resize the images to have the same height
        scale = h1 / h2 if h1 < h2 else h2 / h1
        img1 = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
        img2 = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
    return np.concatenate((img1, img2), axis=1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the file to disk
            filename = file.filename
            file.save(os.path.join('uploads', filename))

            # Read the uploaded image
            img = cv2.imread(os.path.join('uploads', filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply the ESRGAN model to the image
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Apply image editing
            brightness = int(request.form['brightness'])
            contrast = int(request.form['contrast'])
            gamma = float(request.form['gamma'])
            blur = int(request.form['blur'])
            filter_type = request.form['filter']
            flip_horizontal = 'flip_horizontal' in request.form
            flip_vertical = 'flip_vertical' in request.form
            rotate = int(request.form['rotate'])
            scale = float(request.form['scale'])
            crop = 'crop' in request.form
            if crop:
                x1 = int(request.form['x1'])
                y1 = int(request.form['y1'])
                x2 = int(request.form['x2'])
                y2 = int(request.form['y2'])
                img_edit = output[y1:y2, x1:x2]
            else:
                img_edit = output
            img_edit = cv2.cvtColor(img_edit.astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_edit = cv2.convertScaleAbs(img_edit, alpha=gamma, beta=brightness)
            img_edit = cv2.addWeighted(img_edit, 1.0 + contrast / 100.0, img_edit, 0, 0)
            img_edit = cv2.GaussianBlur(img_edit, (blur * 2 + 1, blur * 2 + 1), 0)
            if filter_type == 'Grayscale':
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'Sepia':
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB)
                img_edit = cv2.transform(img_edit, np.array([[0.393, 0.769, 0.189],
                                                             [0.349, 0.686, 0.168],
                                                             [0.272, 0.534, 0.131]]))
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_RGB2BGR)
            elif filter_type == 'Invert':
                img_edit = cv2.bitwise_not(img_edit)
            elif filter_type == 'Sketch':
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
                img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
                img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'Cartoon':
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
                img_edit = cv2.medianBlur(img_edit, 7)
                edges = cv2.Laplacian(img_edit, cv2.CV_8U, ksize=5)
                ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
                img_edit = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            elif filter_type == 'Pencil':
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
                img_edit = cv2.GaussianBlur(img_edit, (21, 21), 0, 0)
                img_edit = cv2.divide(img_edit, 255 - img_edit, scale=256)
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_BGR2GRAY)
                img_edit = cv2.adaptiveThreshold(img_edit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
                img_edit = cv2.cvtColor(img_edit, cv2.COLOR_GRAY2BGR)
            if flip_horizontal:
                img_edit = cv2.flip(img_edit, 1)
            if flip_vertical:
                img_edit = cv2.flip(img_edit, 0)
            if rotate != 0:
                (h, w) = img_edit.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, rotate, scale)
                img_edit = cv2.warpAffine(img_edit, M, (w, h))

            # Merge with another image
            merge_file = request.files.get('merge_file')
            if merge_file:
                merge_img = cv2.imdecode(np.fromstring(merge_file.read(), np.uint8), cv2.IMREAD_COLOR)
                merge_img = cv2.cvtColor(merge_img, cv2.COLOR_RGB2BGR)
                merge_img = cv2.resize(merge_img, (img_edit.shape[1], img_edit.shape[0]))
                img_edit = merge_images(img_edit, merge_img)

            # Save the edited image to disk
            cv2.imwrite(os.path.join('static', 'output.jpg'), img_edit)

            # Render the result page
            return render_template('result.html', filename=filename)

    # Render the upload page
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
