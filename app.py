import io
import os
import json
from PIL import Image
import time
import glob
import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__, static_folder='static')

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
model.eval()


def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
    
    
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()

        try:
           
        
            results = get_prediction(img_bytes)

            # Generate a unique folder name
            folder_name = str(time.time())
            results_dir = os.path.join(app.config['RESULT_FOLDER'], folder_name)

            # Ensure the folder exists
            # os.makedirs(results_dir, exist_ok=True)
            #print("yes")
            # Save the results
            results.save(save_dir=results_dir)

            # Print the contents of the results directory
            print("Directory contents:", os.listdir(results_dir))
            
            # Get the name of the saved image
            image_files = glob.glob(os.path.join(results_dir, '*.jpg'))
            if image_files:
                image_filename = os.path.basename(image_files[0])
                # The image path should now be 'static/timestamp/image_filename'
                image_path = os.path.join(folder_name, image_filename).replace('\\', '/')
            else:
                print("No .jpg files found in directory")
                image_path = ''
            
            image_url = url_for('static', filename=image_path)

            print("Image URL: ", image_url)  # Print the image URL for debugging

            return render_template('Home.html', result_image=image_url)
            #return render_template('result.html', result_image='static/image0.jpg')

        except Exception as e:
            print(str(e))  # Print the exception for debugging
            return render_template('error.html', error=str(e))
        

    return render_template('Home.html')



if __name__ == '__main__':
   
    app.run()
