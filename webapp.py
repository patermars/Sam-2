from flask import Flask, render_template, request, jsonify, send_from_directory
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import subprocess
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from threading import Thread
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from backend_sam2 import main

coordinates_list = []

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
        filename="0.mp4"
        video_path = os.path.join(DOWNLOAD_FOLDER, filename)
        print(video_path)
        counter=0
        while os.path.exists(video_path):
            counter += 1
            filename=f"{counter}.mp4"
            video_path = os.path.join(DOWNLOAD_FOLDER, filename)
        
        video_path = ys.download(output_path=DOWNLOAD_FOLDER, filename=filename)
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
    label=data['label']
    obj_id=data['ann_obj_id']
    coordinates = {'x': x, 'y': y, 'time': current_time,'label':label,'ann_obj_id':obj_id}
    coordinates_list.append(coordinates)
    print(f'Coordinates received - X: {x}, Y: {y}, Time: {current_time}, Label : {label}, Object ID :{obj_id}')
    return jsonify({'status': 'success'})

@app.route('/trim', methods=['POST'])
def trim_video():
    start_time = request.form['startTime']
    end_time = request.form['endTime']
    filename = request.form['filename']
    
    input_path = os.path.join(DOWNLOAD_FOLDER, filename)
    output_filename = f"trimmed_{filename}"
    output_path = os.path.join(DOWNLOAD_FOLDER, output_filename)

    # Handle existing file names
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
        return jsonify({'filename': os.path.basename(output_path)})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():

    print("pass process video")
    data = request.get_json()
    output_path = data.get('output_path')

    print(output_path)
    
    if output_path:
        main(output_path,coordinates_list)
        return jsonify({"message": "Video processed successfully"})
    else:
        return jsonify({"error": "No output path provided"}), 400

def get_frame_rate(video_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vcodec', 'copy',
        '-f', 'null',
        '-'
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, universal_newlines=True)
    for line in result.stderr.split('\n'):
        if 'fps' in line:
            frame_rate = float(line.split('fps')[0].strip().split(' ')[-1])
            return frame_rate
    return None


if __name__ == '__main__':
    app.run(debug=True)
