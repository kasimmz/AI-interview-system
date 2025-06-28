from fileinput import filename
import os
import numpy as np
import fitz
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from flask_cors import CORS  
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import uuid
from datetime import datetime
import sqlite3
from textblob import TextBlob
from collections import Counter
from dotenv import load_dotenv
from json import JSONEncoder
from flask.sessions import SecureCookieSessionInterface
from itsdangerous import URLSafeTimedSerializer
from textblob.download_corpora import download_all
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt',quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger') 
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Optional - downloads all TextBlob corpora
download_all()

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#configuration
ATS_THRESHOLD = 60  # Minimum score to be considered a good match
TOTAL_INTERVIEWS_QUESTIONS = 20 # Total number of interview questions to be asked

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, 
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float16, np.float32, 
                            np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8, 
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, 
                         np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj        

class NumpySerializer:
    """Custom serializer for handling numpy types in Flask sessions"""
    
    def dumps(self, obj):
        converted_obj = convert_numpy_types(obj)
        return json.dumps(converted_obj, cls=NumpyJSONEncoder)
    
    def loads(self, s):
        return json.loads(s)
    
class NumpySessionInterface(SecureCookieSessionInterface):
    def get_signing_serializer(self, app):
        if not app.secret_key:
            return None
        signer_kwargs = dict(
            key_derivation = self.key_derivation,
            digest_method = self.digest_method,
        ) 
        return URLSafeTimedSerializer(
            app.secret_key,
            salt=self.salt,
            serializer=NumpySerializer(),
            signer_kwargs=signer_kwargs
        )

# Set the custom session interface
app.session_interface = NumpySessionInterface()

            
def init_db():
    conn = sqlite3.connect('interview_system.db')
    cursor = conn.cursor()
    
    #Interview sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_sessions (
            session_id TEXT PRIMARY KEY,
            candidate_name TEXT,
            job_title TEXT,
            ats_score REAL,
            status TEXT,
            start_time Text,
            end_time Text,
            overall_score REAL,
            result TEXT                            
        )
    ''')

    #Interview questions table -updated to store questions even without responses    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interview_qa (
            qa_id INTEGER PRIMARY KEY,
            session_id TEXT,
            question_no INTEGER,
            question_text TEXT,       
            question_type TEXT,
            response_text TEXT DEFAULT '',
            response_score REAL DEFAULT 0,
            evaluation_details TEXT DEFAULT '',               
            FOREIGN KEY (session_id) REFERENCES interview_sessions (session_id) 
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting pdf:{str(e)}")
    return text

def cos_similarity(resume_text,job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] *100, 2)

def enhanced_ats_analysis(resume_text, job_desc):
    """Enhanced ATS analysis with detailed scoring"""
    overall_score = float(cos_similarity(resume_text, job_desc))

    #extract skills and keywords
    job_desc_keywords = extract_keywords(job_desc.lower())
    resume_keywords = extract_keywords(resume_text.lower())

    # calculate detailed metrics
    skill_match = float(calculate_skill_match(resume_keywords, job_desc_keywords))
    keyword_density = float(len(set(resume_keywords) & set(job_desc_keywords))) / float(len(set(job_desc_keywords))) * 100

    result = {
        'overall_score': overall_score,
        'skill_match': skill_match,
        'keyword_density': round(keyword_density, 2),
        'threshold_passed': bool(overall_score >= ATS_THRESHOLD),
        'matched_keywords': list(set(resume_keywords) & set(job_desc_keywords)),
        'missing_keywords': list(set(job_desc_keywords) - set(resume_keywords))
    }
    
    return convert_numpy_types(result)

def extract_keywords(text, top_n=20, min_length=3):
    """
    Extract keywords from text with NLP techniques
    
    Args:
        text(str): Input text to extract keywords from
        top_n(int): Number of top keywords to return
        min_length(int): Minimum length of keywords to consider

    Returns:
        list:Top keywords ordered by importance
    """
    #preprocessing
    words = re.findall(r'\b\w{%d,}\b' % min_length, text.lower())
    words = [word for word in words if len(word) >= min_length]

    #enhanced stopword list
    extented_stopwords = {
        "the", "and", "for", "are", "with", "that", "this", "will", "from", 
        "have", "you", "can", "your", "about", "they", "their", "there","which",
        "were", "been", "would", "should", "could", "where", "when", "what","into",
        "from", "than", "them", "these", "those"
    }
    #filter stopwords and get word frequencies
    filtered_words = [word for word in words if word not in extented_stopwords]
    word_freq = Counter(filtered_words)

    #get embeddings for remaining words
    unique_words = list(word_freq.keys())
    if not unique_words:
        return []
    
    #encode words using sentence transformer
    word_embeddings = model.encode(unique_words)

    #calculate document embeddings
    doc_embedding = np.mean(model.encode([text]), axis=0)

    #calculate cosine similarity for each word
    similarities = cosine_similarity(word_embeddings, [doc_embedding]).flatten()

    #combine frequency and semantic relevance
    scores = {}
    for i, word in enumerate(unique_words):
        #weighted score: 50% freq,50% sem
        scores[word] = 0.5 * word_freq[word] + 0.5 * similarities[i]

    #get top keykeywords
    sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [kw[0] for kw in sorted_keywords[:top_n]]

    return top_keywords

def calculate_skill_match(resume_keywords, job_desc_keywords):
    """calculate skill match percentage"""
    if not job_desc_keywords:
        return 0
    matched_skills = len(set(resume_keywords) & set(job_desc_keywords))
    return round((matched_skills / len(job_desc_keywords)) * 100, 2)

class InterviewQuestionGenerator:
    def __init__(self, questions_file = os.path.join("data","questions_ds.json")):
        try:
            with open(questions_file) as f:
                self.question_bank = json.load(f)
        except FileNotFoundError:
            self.question_bank = {
                'technical': [
                    "Tell me about your technical background and expertise.",
                    "How do you stay updated with the latest technologies?",
                    "Describe a challenging technical problem you solved."
                ],
                'behavioral': [
                    "Tell me about yourself.",
                    "Why are you interested in this position?",
                    "Describe a time when you worked in a team."
                ],
                'situational': [
                    "How would you handle a tight deadline?",
                    "What would you do if you disagreed with your manager?",
                    "How would you approach a project with unclear requirements?",
                ],
                'role_specific': [
                    "What specific skills do you bring to this role?",
                    "How do you prioritize tasks in your work?",
                    "What tools or technologies are you most comfortable with?"
                ]
            }
    def generate_questions(self, job_desc, resume_text, num_questions=20):
        """Generate personalized interview questions from dataset"""
        questions = []
        question_types = list(self.question_bank.keys())
        questions_per_type = num_questions // len(question_types)
        
        # Add questions from each category
        for q_type in question_types:
            type_questions = self.question_bank[q_type][:questions_per_type]
            for q in type_questions:
                questions.append({
                    'question': q,
                    'type': q_type,
                    'number': len(questions) + 1
            })
        
        # Fill remaining slots if needed
        remaining = num_questions - len(questions)
        if remaining > 0:
            for i in range(remaining):
                q_type = question_types[i % len(question_types)]
                available_questions = [
                    q for q in self.question_bank[q_type] 
                    if q not in [x['question'] for x in questions]
                ]
                if available_questions:
                    questions.append({
                        'question': available_questions[0],
                        'type': q_type,
                        'number': len(questions) + 1
                    })
        
        return questions[:num_questions]       

class ResponseEvaluator:
    def __init__(self):
        #ensures nltk data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def evaluate_response(self, question, answer,question_type):
        """evaluate interview response"""
        try:
            if not answer or answer.strip() == "":
                return{
                    'technical_accuracy': 0.0,
                    'communication_clarity': 0.0,
                    'relevance': 0.0,
                    'completeness': 0.0,
                    'overall_score': 0.0,
                    'word_count': 0,
                    'feedback': "No response provided"
                }
            word_count = len(answer)
            sentence_count = 1 
            try:
                #text analysis
                blob = TextBlob(answer)
                sentence_count = len(blob.sentences)
                if sentence_count==0:
                    sentence_count=1
            except Exception as blob_error:
                print(f"TextBlob Error: {str(blob.sentences)}")
                #fallback: count sentences by periods
                sentence_count = max(1, answer.count('.') + answer.count('!') +answer.count('?'))

            #basic scoring algorithm
            scores = {}

            #communication clarity
            if word_count < 20:
                scores['communication_clarity'] = 40
            elif word_count < 50:
                scores['communication_clarity'] = 60
            elif word_count < 100:
                scores['communication_clarity'] = 80
            else:
                scores['communication_clarity'] = 90

            #relevence 
            try:
                question_words = set(question.lower().split())
                answer_words = set(answer.lower().split())
                relevance_ratio = len(question_words & answer_words) / len(question_words) if question_words else 0
                scores['relevance'] = min(90, relevance_ratio * 100 + 30)
            except Exception:
                scores['relevance'] = 50  # Default if error occurs

            #completeness
            if word_count>=30 and sentence_count>=2:
                scores['completeness'] = 80
            elif word_count>=15:
                scores['completeness'] = 60
            else:
                scores['completeness'] = 40

            #technical accuracy
            try:
                blob = TextBlob(answer)
                sentiment = blob.sentiment.polarity
                if sentiment > 0:
                    scores['technical_accuracy'] = 70 + (sentiment * 20)
                else:
                    scores['technical_accuracy'] = 50
            except Exception:
                #fallback: base score on word count and question type
                if question_type == 'technical':
                    scores['technical_accuracy'] = min(80, 40 + (word_count* 0.5))        
                else:
                    scores['technical_accuracy'] = 65
        
             #overall score
            overall_score = sum(scores.values()) / len(scores)

            result = {
                'technical_accuracy': float(round(scores['technical_accuracy'], 2)),
                'communication_clarity': float(round(scores['communication_clarity'], 2)),
                'relevance': float(round(scores['relevance'], 2)),
                'completeness': float(round(scores['completeness'], 2)),
                'overall_score': float(round(overall_score, 2)),
                'word_count': int(word_count),
                'feedback': self.generate_feedback(scores, word_count)
            }
            return convert_numpy_types(result)
    
        except Exception as e:
            print(f"Error evaluating response: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return default scores in case of error
            return {
                'technical_accuracy': 50.0,
                'communication_clarity': 50.0,
                'relevance': 50.0,
                'completeness': 50.0,
                'overall_score': 50.0,
                'word_count': len(answer.split()) if answer else 0,
                'feedback': "Response evaluted with basic scoring due to processing error."
            }

    def generate_feedback(self, scores, word_count):
        """Generate feedback based on scores"""
        try:
            feedback = []
            if scores['communication_clarity'] < 60:
                feedback.append("Try to provide more detailed answers with clear explanations.")
            if scores['relevance'] < 60:
                feedback.append("Focus more on directly answering the question asked.")
            if scores['completeness'] < 60:
                feedback.append("Provide more comprehensive answers with examples.")
            if word_count < 30:
                feedback.append("Considerproviding more detailed explanations.")
            if not feedback:
                feedback.append("Good job! Keep up the good work.")

            return " ".join(feedback)
        except Exception:
            return "Response received and evaluated."

class InterviewReportGenerator:
    def __init__(self):
        pass

    def generate_report(self, session_id):
        """Generate detailed interview report"""
        conn = sqlite3.connect('interview_system.db')
        cursor = conn.cursor()

        # Get session data
        cursor.execute('''
            SELECT * FROM interview_sessions WHERE session_id = ?
        ''', (session_id,))
        session_data = cursor.fetchone()

        #get interview questions and responses
        cursor.execute('''
            SELECT * FROM interview_qa WHERE session_id = ? ORDER BY question_no
        ''', (session_id,))
        qa_data = cursor.fetchall()

        conn.close()

        if not session_data:
            return None

        #calculate performance metrics
        total_score = 0
        category_scores = {'techniacal':[], 'behavioral':[], 'situational':[], 'role_specific':[]}  

        detailed_qa = []
        for qa in qa_data:
            qa_dict = {
                'question_no': qa[2],
                'question': qa[3],
                'question_type': qa[4],
                'response': qa[5],
                'score': float(qa[6]) if qa[6] else 0,
                'evaluation': json.loads(qa[7]) if qa[7] else {}
            }
            detailed_qa.append(qa_dict)

            if qa[6]:
                total_score += qa[6]
                if qa[4] in category_scores:
                    category_scores[qa[4]].append(qa[6])
        
        avg_score = total_score / len(qa_data) if qa_data else 0
        
        #calculate category averages
        category_averages = {}
        for category, scores in category_scores.items():
            category_averages[category] = sum(scores) / len(scores) if scores else 0

        #determine pass or fail
        result = "PASSED" if avg_score >= 70 else "FAILED"

        # generate recommendations
        recommedations = self.generate_recommendations(category_averages, detailed_qa)

        report = {
            'session_info': {
                'session_id': session_data[0],
                'candidate_name': session_data[1],
                'job_title': session_data[2],
                'ats_score': session_data[3],
                'status': session_data[4],
                'interview_date': session_data[5],
                'duration': self.calculate_duration(session_data[5], session_data[6])
            },
            'results':{
                'overall_result': result,
                'overall_score': round(avg_score, 2),
                'ats_score': session_data[3],
                'questions_answered': len(qa_data)
            },
            'performance_breakdown': {
                'technical_skills': round(category_averages.get('technical', 0), 2),
                'behavioral': round(category_averages.get('behavioral', 0), 2),
                'situational': round(category_averages.get('situational', 0), 2),
                'role_specific': round(category_averages.get('role_specific', 0), 2)
            },
            'detailed_qa': detailed_qa,
            'recommendations': recommedations,
            'strengths': self.identify_strengths(category_averages),
            'improvement_areas': self.identify_improvements(category_averages)
        }

        return convert_numpy_types(report)
    
    def calculate_duration(self, start_time, end_time):
        """Calculate interview duration"""
        if not start_time or not end_time:
            return "N/A"
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            return str(duration).split('.')[0]  # Return as HH:MM:SS
        except:
            return "N/A"

    def generate_recommendations(self, category_scores, qa_data):
        """Generate personalized recommendations based on performance"""
        recommendations = []
        for category, score in category_scores.items():
            if score < 60:
                if category == 'technical':
                    recommendations.append(f"focus on strengthening your techical skills and practice explaining complex concepts clearly.")  
                elif category == 'behavioral':
                    recommendations.append("Consider practicing common behavioral questions and structuring your answers using the STAR method (Situation, Task, Action, Result).")       
                elif category == 'situational':
                    recommendations.append("Work on your problem-solving skills and practice situational questions to improve your response strategies.")
                elif category == 'role_specific':
                    recommendations.append("Research more about the company, role, and industry trends.")

        if not recommendations:
            recommendations.append("Great job! Keep up the good work and continue to refine your skills.")

        return recommendations

    def identify_strengths(self, category_scores):
        """Identify candidate strengths"""
        strengths = []
        for category, score in category_scores.items():
            if score >= 75:
                strengths.append(f"Strong {category.replace('_', ' ')} skills")
        
        if not strengths:
            strengths.append("Shows potential for growth")
        
        return strengths
    
    def identify_improvements(self, category_scores):
        """Identify areas for improvement"""
        improvements = []
        for category, score in category_scores.items():
            if score < 65:
                improvements.append(f"{category.replace('_', ' ').title()} skills need development")
        
        return improvements

# Initialize helper classes
question_generator = InterviewQuestionGenerator()
response_evaluator = ResponseEvaluator()
report_generator = InterviewReportGenerator()  

def get_db_connection():
    """Get a database connection"""
    try:
        conn = sqlite3.connect('interview_system.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise

def execute_db_query(query, params=None, fetch=False):
    """Execute a database query with error handling"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if fetch:
            result = cursor.fechall()
            return result
        else:
            conn.commit()
            return cursor.rowcount
    except Exception as e:
        print(f"Database query error:{str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()        


@app.route("/", methods = ['GET', 'POST'])    
def index():
    if request.method == 'POST':
        if "resume" not in request.files:
            return "NO File was uploaded.", 400
        file = request.files["resume"]
        job_desc = request.form["job_desc"]
        candidate_name = request.form.get("candidate_name", "Unknown")

        if file.filename == "" or job_desc.strip() == "":
            return "Invalid Input", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        ats_analysis = enhanced_ats_analysis(resume_text, job_desc)

        # Store interview session in database AND session
        session['candidate_name'] = str(candidate_name)
        session['job_desc'] = str(job_desc)  # Keep this in session for now
        session['resume_text'] = str(resume_text)  # Keep this in session for now
        session['ats_analysis'] = convert_numpy_types(ats_analysis)

        return render_template("index.html",
                               score=float(ats_analysis['overall_score']),
                               analysis=convert_numpy_types(ats_analysis),
                               threshold=ATS_THRESHOLD)
    
    return render_template("index.html", score=None)

@app.route("/start_interview")
def start_interview():
    """start the interview phase"""
    if 'ats_analysis' not in session or not session['ats_analysis']['threshold_passed']:
        return redirect(url_for('index'))
    
    # Check if we have the required data in session
    if 'job_desc' not in session or 'resume_text' not in session:
        return redirect(url_for('index'))
    
    # Create interview session
    session_id = str(uuid.uuid4())
    session['interview_session_id'] = session_id

    # Generate questions BEFORE removing data from session
    questions = question_generator.generate_questions(
        session['job_desc'],
        session['resume_text'],
        TOTAL_INTERVIEWS_QUESTIONS
    )

    # Store questions in database
    conn = sqlite3.connect('interview_system.db')
    cursor = conn.cursor()
    
    # Save session to database
    cursor.execute('''
        INSERT INTO interview_sessions
        (session_id, candidate_name, job_title, ats_score, status, start_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, session['candidate_name'], "Job Position",
          session['ats_analysis']['overall_score'], "IN_PROGRESS",
          datetime.now().isoformat()))
    
    # Save questions to database
    for i, question in enumerate(questions):
        cursor.execute('''
            INSERT INTO interview_qa
            (session_id, question_no, question_text, question_type, response_text, response_score, evaluation_details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, question['number'], question['question'], question['type'], '', 0, ''))
    
    conn.commit()
    conn.close()

    # NOW remove large data from session to reduce cookie size
    session['current_question'] = 0
    session.pop('resume_text', None)
    session.pop('job_desc', None)

    return render_template("interview.html", 
                           question=questions[0], 
                           question_num=1,
                           total_questions=len(questions))

@app.route("/submit_answer", methods=['POST'])
def submit_answer():
    """Submit interview answer and evaluate"""
    try:
        print("=== Submit Answer Route Called ===")
        print(f"Session keys: {list(session.keys())}")
        
        if 'interview_session_id' not in session:
            print("Error: Interview session not found in session")
            return jsonify({'error': 'Interview session not found'}), 400
        
        data = request.get_json()
        if not data:
            print("Error: No JSON data received")
            return jsonify({'error': 'No data received'}), 400
            
        answer = data.get('answer', '').strip()
        print(f"Received answer: {answer[:100]}...")  # Log first 100 chars
        
        current_q_index = session.get('current_question', 0)
        
        #get questions from db instead of session
        conn = sqlite3.connect('interview_system.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT question_no, question_text, question_type
            FROM interview_qa
            WHERE session_id = ?
            ORDER BY question_no
        ''', (session['interview_session_id'],))
        questions_data = cursor.fetchall()
        conn.close()

        if not questions_data:
            print("Error: No interview questions found in session")
            return jsonify({'error': 'Interview questions not found'}), 400
            
        if current_q_index >= len(questions_data):
            print("Error: Question index out of range")
            return jsonify({'error': 'Invalid question index'}), 400
            
        current_question_data= questions_data[current_q_index]
        current_question = {
            'number': current_question_data[0],
            'question': current_question_data[1],
            'type': current_question_data[2]
        }

        print(f"Processing question {current_q_index + 1}: {current_question['question'][:50]}...")  # Log first 50 chars
              

        # Evaluate response with error handling
        try:
            evaluation = response_evaluator.evaluate_response(
                current_question['question'],
                answer,
                current_question['type']
            )
            print(f"Evaluation completed: {evaluation.get('overall_score', 'N/A')}")
        except Exception as eval_error:
            print(f"Error during evaluation: {str(eval_error)}")
            return jsonify({'error': 'Error evaluating response'}), 500

        # Save response to database with error handling
        try:
            conn = sqlite3.connect('interview_system.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE interview_qa
                SET response_text = ?, response_score = ?, evaluation_details = ?
                WHERE session_id = ? AND question_no = ? 
            ''', (
                answer, 
                float(evaluation['overall_score']),
                json.dumps(convert_numpy_types(evaluation)),
                session['interview_session_id'],
                current_question['number']
            ))
            conn.commit()
            conn.close()
            print("Response saved to database successfully")
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Database error'}), 500

        # Move to next question
        session['current_question'] = current_q_index + 1

        if session['current_question'] < len(questions_data):
            # More questions remaining
            next_question = questions_data[session['current_question']]
            next_question = {
                'number': next_question[0],
                'question': next_question[1],
                'type': next_question[2]
            }
            response_data = {
                'status': 'continue',
                'next_question': next_question,
                'question_num': session['current_question'] + 1,
                'evaluation': convert_numpy_types(evaluation)
            }
            print(f"Returning continue response for question {session['current_question'] + 1}")
            return jsonify(response_data)
        else:
            # Interview complete
            response_data = {
                'status': 'complete',
                'evaluation': convert_numpy_types(evaluation)
            }
            print("Interview completed")
            return jsonify(response_data)
            
    except Exception as e:
        print(f"Unexpected error in submit_answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    

@app.route("/complete_report")
def complete_report():
    """Complete the iterview and generate report"""
    if 'interview_session_id' not in session:
        return redirect(url_for('index'))

    session_id = session['interview_session_id']
    
    #update session status
    conn = sqlite3.connect('interview_system.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE interview_sessions
        SET status = ?, end_time = ?
        WHERE session_id = ?
    ''', ("COMPLETED", datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()

    #generate report
    report = report_generator.generate_report(session_id)

    #clear session data
    session.clear() 

    return render_template("report.html", report=report)

@app.route("/view_report/<session_id>")
def view_report(session_id):
    """View detailed interview report"""
    report = report_generator.generate_report(session_id)
    if not report:
        return "Report not found", 404

    return render_template("report.html", report=report)

@app.route("/debug/session")
def debug_session():
    """Debug route to check session data"""
    return jsonify({
        'session_keys' : list(session.keys()),
        'has_interview_session': 'interview_session_id' in session,
        'has_questions' : 'interview_questions' in session,
        'current_question' : session.get('current_question', 'NOT set'),
        'questions_count' : len(session.get('interview_questions', [])),
    })
@app.route("/test_connection")
def test_connection():
    """Test route to verify server is working"""
    return jsonify({
        'status': 'Server is running',
        'timestamp': datetime.now().isoformat()
    })
   
if __name__ =="__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)
