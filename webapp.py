from flask import Flask, render_template, request, jsonify, send_from_directory
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import subprocess

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

@app.route('/coordinates', methods=['POST'])
def log_coordinates():
    data = request.json
    x = data['x']
    y = data['y']
    current_time = data['currentTime']
    print(f'Coordinates received - X: {x}, Y: {y}, Time: {current_time}')
    return jsonify({'status': 'success'})

@app.route('/trim', methods=['POST'])
def trim_video():
    start_time = request.form['startTime']
    end_time = request.form['endTime']
    filename = request.form['filename']
    
    input_path = os.path.join(DOWNLOAD_FOLDER, filename)
    output_filename = f"trimmed_{filename}"
    output_path = os.path.join(DOWNLOAD_FOLDER, output_filename)

    input_path = os.path.join(DOWNLOAD_FOLDER, filename)
    output_filename = f"trimmed_{filename}"
    output_path = os.path.join(DOWNLOAD_FOLDER, output_filename)

    base, ext = os.path.splitext(output_path)
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base}({counter}){ext}"
        counter += 1

    
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', start_time,
        '-to', end_time,
        '-c', 'copy',
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        return jsonify({'filename': output_filename})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
