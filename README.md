
# Potato Leaf Disease Classification

This is a potato leaf disease classification application that uses a Convolutional Neural Network (CNN) model to identify diseases in potato leaves. The application includes a FastAPI backend for model inference and a React frontend for image upload and prediction display.

## How to Run the Project

To run the project locally, follow these steps:

### Clone the Repository

```bash
git clone https://github.com/your-username/potato-leaf-disease-classification.git
cd potato-leaf-disease-classification
```

### Backend Setup

1. Navigate to the `api` directory:

    ```bash
    cd api
    ```

2. Install the dependencies:

    ```bash
    pip install -r ../AI_Project/requirements.txt
    ```

3. Start the FastAPI server:

    ```bash
    uvicorn main:app --reload
    ```

   The API will be available at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the `frontend-ai` directory:

    ```bash
    cd ../frontend-ai
    ```

2. Install the dependencies:

    ```bash
    npm install
    ```

3. Start the development server:

    ```bash
    npm start
    ```

   The React app will be available at `http://localhost:3000`.

## API Used

This project uses a custom-trained TensorFlow model for potato leaf disease classification. The model is hosted within the FastAPI backend.

## Additional Notes

- The project is built using FastAPI for the backend and React for the frontend.
- The backend handles image upload, preprocessing, and disease prediction.
- The frontend allows users to upload images of potato leaves and displays the predicted disease class and confidence.
- The trained model is included in the repository and loaded by the FastAPI backend.

---

### Project Structure

```plaintext
.
├── AI_Project
│   ├── PlantVillage
│   │   └── ... # Images of potato leaves categorized into different disease classes
│   ├── models
│   │   └── ... # Trained model files
│   ├── training.py # Script for training the model
│  
├── api
│   ├── main.py # FastAPI backend for model inference
│   └── main-tfserving.py # Alternative FastAPI backend using TensorFlow Serving
    └── requirements.txt # Dependencies for the AI project
├── frontend-ai
│   ├── node_modules
│   │   └── ... # Node.js modules
│   ├── public
│   │   ├── favicon.ico
│   │   ├── index.html # HTML file for the React application
│   │   ├── logo192.png
│   │   ├── logo512.png
│   │   ├── manifest.json
│   │   └── robots.txt
│   ├── src
│   │   ├── App.css # CSS for the React application
│   │   ├── App.js # Main React component
│   │   ├── App.test.js
│   │   ├── bg.png # Background image for the React application
│   │   ├── index.css
│   │   ├── index.js # Entry point for the React application
│   │   ├── logo.svg
│   │   ├── reportWebVitals.js
│   │   └── setupTests.js
│   ├── .gitignore
│   ├── package-lock.json
│   └── package.json
└── README.md # Project documentation



---

You can copy this content into your `README.md` file when uploading the project to GitHub.
