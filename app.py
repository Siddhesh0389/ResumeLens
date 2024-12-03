from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import re

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('resume_screening_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Manually define job categories
job_categories = {
    0: 'Data Scientist',
    1: 'Software Engineer',
    2: 'Web Developer',
    3: 'AI/ML Engineer',
    4: 'Database Administrator',
    19: 'Network Engineer'  # Example mapping
}

# Manually define suggested skills for each job category
suggested_skills_map = {
    'Data Scientist': ['Python', 'Machine Learning', 'Data Analysis', 'Pandas', 'Numpy'],
    'Software Engineer': ['Java', 'C++', 'Algorithms', 'Data Structures'],
    'Web Developer': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js'],
    'AI/ML Engineer': ['Python', 'TensorFlow', 'Keras', 'NLP', 'Deep Learning'],
    'Database Administrator': ['SQL', 'Database Design', 'Oracle', 'NoSQL'],
    'Network Engineer': ['Cisco', 'Network Security', 'Troubleshooting', 'TCP/IP']
}

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Function to process uploaded resume
def process_uploaded_resume(file_path):
    extension = file_path.rsplit('.', 1)[1].lower()
    if extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif extension == 'docx':
        return extract_text_from_docx(file_path)

# Helper function to calculate resume accuracy
def calculate_resume_accuracy(resume_text, job_description):
    resume_vector = vectorizer.transform([resume_text])
    job_description_vector = vectorizer.transform([job_description])
    similarity_score = cosine_similarity(resume_vector, job_description_vector)[0][0]
    return round(similarity_score * 100, 2)

# Helper function to extract skills from resume text
def extract_skills_from_resume(resume_text):
    # Simple regex to find key skills
    skills = re.findall(r'(Python|Java|SQL|HTML|CSS|JavaScript|Machine Learning|Deep Learning|Network Security)', resume_text, re.IGNORECASE)
    return list(set(skills))

# Suggest missing skills based on job category
def suggest_skills(resume_skills, job_category):
    job_skills = suggested_skills_map.get(job_category, [])
    missing_skills = [skill for skill in job_skills if skill not in resume_skills]
    return missing_skills

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def about():
    return render_template('second.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the job description input
        job_description = request.form['job_description']
        
        # Check if a file is uploaded
        if 'resume' not in request.files or request.files['resume'].filename == '':
            return redirect(url_for('index'))
        
        file = request.files['resume']
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ['pdf', 'docx']:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            
            # Process the uploaded resume and extract text
            resume_text = process_uploaded_resume(file_path)

            # Log the extracted text for debugging
            print("Extracted Resume Text:", resume_text)

            # Ensure the extracted text is cleaned (strip extra whitespaces, etc.)
            resume_text = resume_text.strip()
            
            if len(resume_text) == 0:
                return "Error: Could not extract text from the resume. Please upload a valid resume."

            # Predict the job category using the machine learning model
            resume_vector = vectorizer.transform([resume_text])
            predicted_job_category_index = model.predict(resume_vector)[0]
            predicted_job_category = job_categories.get(predicted_job_category_index, 'Unknown')
            
            # Log the prediction for debugging
            print("Predicted Job Category:", predicted_job_category)
            
            # Calculate resume accuracy
            accuracy_score = calculate_resume_accuracy(resume_text, job_description)
            
            # Extract and suggest missing skills
            resume_skills = extract_skills_from_resume(resume_text)
            suggested_skills = suggest_skills(resume_skills, predicted_job_category)
            
            # Render the results
            return render_template(
                'result.html',
                accuracy=accuracy_score,
                resume_skills=resume_skills,
                missing_skills=suggested_skills
            )
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
