from flask import Flask, request, jsonify, send_file, render_template
import torch
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model
def load_model(model_path):
    try:
        # Import your model class
        from init import DeblurGeneratorMobileViT

        # Initialize model
        model = DeblurGeneratorMobileViT(image_size=(256, 256)).to(device)

        # Load weights with weights_only=True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Transform for input images
transform = transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor()
])

def process_image(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Convert output tensor to image
        output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())

        # Save the result
        result_path = os.path.join(
        app.config['RESULTS_FOLDER'],
        f"deblurred_{os.path.basename(image_path)}"
        )
        output_image.save(result_path)

        return result_path
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Initialize model
model = load_model('C:/Users/pc/BlurFix/model/generator.pth')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/deblur', methods=['POST'])
def deblur_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_path = process_image(filepath)

        if result_path is None:
            return jsonify({'error': 'Error processing image'}), 500

        return send_file(result_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)