"""
database.py - Persistence Layer for ParkinsonsSpeech.AI
Supports SQLite (local dev) and PostgreSQL (production) via SQLAlchemy.
Models: User, Prediction, Feedback
"""

import os
import uuid
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(100), unique=True, nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    sample_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=True)
    
    # Core stored metrics
    feature_vector = db.Column(db.JSON, nullable=False)
    ml_probability = db.Column(db.Float, nullable=False)
    acoustic_index = db.Column(db.Float, nullable=False)
    biomarkers = db.Column(db.JSON, nullable=True)
    risk_score = db.Column(db.Float, nullable=False)
    risk_label = db.Column(db.String(50), nullable=False)
    model_version = db.Column(db.String(50), default="v1.0.0-calibrated-xgboost")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "ml_probability": round(self.ml_probability, 4),
            "acoustic_index": round(self.acoustic_index, 4),
            "biomarkers": self.biomarkers,
            "risk_score": round(self.risk_score, 4),
            "risk_label": self.risk_label,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "feature_vector": self.feature_vector
        }


class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sample_id = db.Column(db.String(36), db.ForeignKey('predictions.sample_id'), nullable=False)
    user_rating = db.Column(db.Integer, nullable=True)
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "sample_id": self.sample_id,
            "user_rating": self.user_rating,
            "comments": self.comments,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


def init_database(app):
    db_path = os.path.join(app.root_path, 'parkinson.db')
    default_uri = f"sqlite:///{db_path}"
    
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', default_uri)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        print(f"[DATABASE] Initialized persistence layer: {app.config['SQLALCHEMY_DATABASE_URI']}")
