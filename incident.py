"""
incident.py
Saves incident video clips, event data, hash, and timestamp to a local SQLite database.
"""
import sqlite3
import hashlib
import time
import cv2
import os
import collections

class IncidentLogger:
    def __init__(self, db_path='incidents.db', buffer_size=60):
        self.conn = sqlite3.connect(db_path)
        self.create_table()
        self.frame_buffer = collections.deque(maxlen=buffer_size)  # Buffer for last N frames
        self.buffer_size = buffer_size

    def create_table(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_data TEXT,
            video_path TEXT,
            hash TEXT
        )''')
        self.conn.commit()

    def buffer_frame(self, frame):
        self.frame_buffer.append(frame.copy())

    def log_incident(self, event_data, video_path, fps=30):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        hash_val = hashlib.sha256((event_data + timestamp).encode()).hexdigest()
        # Save buffered video
        if self.frame_buffer:
            h, w = self.frame_buffer[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            for f in self.frame_buffer:
                out.write(f)
            out.release()
        c = self.conn.cursor()
        c.execute('INSERT INTO incidents (timestamp, event_data, video_path, hash) VALUES (?, ?, ?, ?)',
                  (timestamp, event_data, video_path, hash_val))
        self.conn.commit()
