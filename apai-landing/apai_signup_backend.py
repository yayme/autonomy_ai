
import os
import psycopg2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Get the database URL from Render environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')

# Ensure table exists
def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    email = request.json.get('email')
    if not email:
        return jsonify({'success': False, 'message': 'No email provided.'}), 400
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute('INSERT INTO emails (email) VALUES (%s) ON CONFLICT DO NOTHING', (email,))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Thank you! You will be notified.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/emails')
def show_emails():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute('SELECT email, created_at FROM emails ORDER BY created_at DESC')
    rows = cur.fetchall()
    cur.close()
    conn.close()
    emails_html = '\n'.join([f"{email} ({created_at})" for email, created_at in rows])
    return f"<pre>{emails_html}</pre>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
