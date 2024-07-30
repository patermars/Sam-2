from flask import Flask, render_template, request, send_from_directory, jsonify
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os

app = Flask(__name__)

DOWNLOAD_FOLDER = 'downloads'
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/download', methods=['POST'])
def download_video():
    video_url = request.form['videoUrl']
    try:
        yt = YouTube(video_url, on_progress_callback=on_progress)
        ys = yt.streams.get_highest_resolution()
        video_path = ys.download(DOWNLOAD_FOLDER)
        filename = os.path.basename(video_path)
        return jsonify({'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
