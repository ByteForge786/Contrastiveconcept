import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler
import json
from tqdm import tqdm
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_dir: str = "logs"):
    """Setup detailed logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        f"{log_dir}/attribute_classifier.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

@dataclass
class Config:
    """Configuration for the attribute classifier"""
    # Model settings
    model_name: str = 'all-MiniLM-L6-v2'
    max_seq_length: int = 512  # Maximum sequence length
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    
    # Training settings
    num_positives: int = 4  # Number of positive pairs per anchor
    num_negatives: int = 8  # Number of negative pairs per anchor
    temperature: float = 0.07
    validation_split: float = 0.1
    
    # Performance settings
    num_workers: int = 4
    max_threads: int = 8
    
    # Paths
    model_save_path: str = 'models/attribute-classifier'
    cache_dir: str = 'cache'
    log_dir: str = 'logs'
    metrics_dir: str = 'metrics'
    
    # Template
    template: str = "In the domain of {domain}, this attribute represents {concept} which is defined as {definition}"

    def __post_init__(self):
        """Create necessary directories"""
        for directory in [self.model_save_path, self.cache_dir, 
                         self.log_dir, self.metrics_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

class ContrastivePair:
    """Container for text-template pairs"""
    def __init__(self, text: str, template: str, label: str):
        self.text = text
        self.template = template
        self.label = label

    def truncate(self, max_length: int) -> None:
        """Truncate text and template to max_length tokens"""
        # Simple word-based truncation
        self.text = ' '.join(self.text.split()[:max_length])
        self.template = ' '.join(self.template.split()[:max_length])

class ContrastiveBatch:
    """Container for a complete training batch"""
    def __init__(self, anchor: ContrastivePair):
        self.anchor = anchor
        self.positives: List[ContrastivePair] = []
        self.negatives: List[ContrastivePair] = []

    def add_positive(self, positive: ContrastivePair):
        self.positives.append(positive)

    def add_negative(self, negative: ContrastivePair):
        self.negatives.append(negative)

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with separate embedding"""
    def __init__(self, 
                 attribute_mappings: Dict[str, List[str]], 
                 template_mappings: Dict[str, str],
                 config: Config,
                 phase: str = "train"):
        self.attribute_mappings = attribute_mappings
        self.template_mappings = template_mappings
        self.config = config
        self.phase = phase
        self.labels = list(attribute_mappings.keys())
        
        logger.info(f"Initialized {phase} dataset with {len(self.labels)} labels")
        self._log_dataset_stats()

    def _log_dataset_stats(self):
        """Log dataset statistics"""
        stats = {
            "total_labels": len(self.labels),
            "total_attributes": sum(len(attrs) for attrs in self.attribute_mappings.values()),
            "attributes_per_label": {
                label: len(attrs) for label, attrs in self.attribute_mappings.items()
            }
        }
        logger.info(f"{self.phase} dataset stats: {json.dumps(stats, indent=2)}")

    def __len__(self):
        return sum(len(attrs) for attrs in self.attribute_mappings.values())

    def __getitem__(self, idx) -> ContrastiveBatch:
        # Select anchor
        anchor_label = random.choice(self.labels)
        anchor_text = random.choice(self.attribute_mappings[anchor_label])
        anchor_template = self.template_mappings[anchor_label]
        
        # Create anchor pair
        anchor = ContrastivePair(anchor_text, anchor_template, anchor_label)
        batch = ContrastiveBatch(anchor)
        
        # Get positive examples (same label)
        positive_texts = [
            text for text in self.attribute_mappings[anchor_label]
            if text != anchor_text
        ]
        if len(positive_texts) < self.config.num_positives:
            positive_texts = positive_texts * (self.config.num_positives // len(positive_texts) + 1)
        positive_texts = random.sample(positive_texts, self.config.num_positives)
        
        for pos_text in positive_texts:
            positive = ContrastivePair(pos_text, anchor_template, anchor_label)
            batch.add_positive(positive)
            
        # Get negative examples (different labels)
        other_labels = [l for l in self.labels if l != anchor_label]
        for _ in range(self.config.num_negatives):
            neg_label = random.choice(other_labels)
            neg_text = random.choice(self.attribute_mappings[neg_label])
            neg_template = self.template_mappings[neg_label]
            
            negative = ContrastivePair(neg_text, neg_template, neg_label)
            batch.add_negative(negative)
            
        return batch

class ImprovedInfoNCE(nn.Module):
    """InfoNCE loss with separate embedding for each text"""
    def __init__(self, model: SentenceTransformer, temperature: float = 0.07):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def encode_pair(self, pair: ContrastivePair) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text and template separately"""
        text_embedding = self.model.encode(
            pair.text, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        template_embedding = self.model.encode(
            pair.template, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return text_embedding, template_embedding

    def compute_similarity(self, 
                         anchor_embeddings: Tuple[torch.Tensor, torch.Tensor],
                         pair_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute similarity between two pairs of embeddings"""
        anchor_text_emb, anchor_template_emb = anchor_embeddings
        pair_text_emb, pair_template_emb = pair_embeddings
        
        text_sim = F.cosine_similarity(anchor_text_emb.unsqueeze(0), 
                                     pair_text_emb.unsqueeze(0))
        template_sim = F.cosine_similarity(anchor_template_emb.unsqueeze(0),
                                         pair_template_emb.unsqueeze(0))
        
        return (text_sim + template_sim) / 2

    def forward(self, batches: List[ContrastiveBatch]) -> torch.Tensor:
        total_loss = 0
        
        for batch in batches:
            # Encode anchor
            anchor_embeddings = self.encode_pair(batch.anchor)
            
            # Encode positives and compute similarities
            positive_similarities = []
            for positive in batch.positives:
                pos_embeddings = self.encode_pair(positive)
                similarity = self.compute_similarity(anchor_embeddings, pos_embeddings)
                positive_similarities.append(similarity)
                
            # Encode negatives and compute similarities
            negative_similarities = []
            for negative in batch.negatives:
                neg_embeddings = self.encode_pair(negative)
                similarity = self.compute_similarity(anchor_embeddings, neg_embeddings)
                negative_similarities.append(similarity)
            
            # Combine similarities
            pos_similarities = torch.stack(positive_similarities) / self.temperature
            neg_similarities = torch.stack(negative_similarities) / self.temperature
            
            all_similarities = torch.cat([pos_similarities, neg_similarities])
            labels = torch.zeros(len(all_similarities), dtype=torch.long)
            labels[:len(pos_similarities)] = 1
            
            # Compute loss
            loss = F.cross_entropy(all_similarities.unsqueeze(0), labels.unsqueeze(0))
            total_loss += loss
            
        return total_loss / len(batches)

class AttributeClassifier:
    """Main classifier using contrastive learning"""
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        
    def train(self, 
             train_dataset: ContrastiveDataset,
             val_dataset: ContrastiveDataset):
        """Train the model"""
        logger.info("Starting training...")
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        # Initialize loss
        loss_fn = ImprovedInfoNCE(self.model, self.config.temperature)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_dataloader, loss_fn)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validation phase
            self.model.eval()
            val_loss = self._validate_epoch(val_dataloader, loss_fn)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best')
                
        # Save final model
        self.save_model('final')
        logger.info("Training completed")
        
    def _train_epoch(self, 
                    dataloader: DataLoader,
                    loss_fn: ImprovedInfoNCE) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                loss = loss_fn(batch)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return total_loss / num_batches
        
    def _validate_epoch(self, 
                       dataloader: DataLoader,
                       loss_fn: ImprovedInfoNCE) -> float:
        """Validate for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as pbar:
                for batch in pbar:
                    loss = loss_fn(batch)
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
        return total_loss / num_batches
    
    def save_model(self, suffix: str):
        """Save model with given suffix"""
        save_path = f"{self.config.model_save_path}/model_{suffix}"
        self.model.save(save_path)
        logger.info(f"Saved model to {save_path}")
        
    def load_model(self, suffix: str):
        """Load model with given suffix"""
        load_path = f"{self.config.model_save_path}/model_{suffix}"
        self.model = SentenceTransformer(load_path)
        logger.info(f"Loaded model from {load_path}")
        
    def predict(self, text: str, templates: Dict[str, str]) -> List[Dict[str, Union[str, float]]]:
        """Predict for a single text input"""
        text_embedding = self.model.encode([text])[0]
        
        similarities = {}
        for label, template in templates.items():
            template_embedding = self.model.encode([template])[0]
            similarity = F.cosine_similarity(
                torch.tensor(text_embedding).unsqueeze(0),
                torch.tensor(template_embedding).unsqueeze(0)
            ).item()
            similarities[label] = similarity
            
        # Sort by similarity
        sorted_predictions = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Format predictions
        predictions = []
        for label, confidence in sorted_predictions[:3]:
            domain, concept = label.split('-')
            predictions.append({
                'domain': domain,
                'concept': concept,
                'confidence': confidence
            })
            
        return predictions

def main():
    """Example usage"""
    # Initialize config
    config = Config()
    
    # Load data
    attributes_df = pd.read_csv("data/attributes.csv")
    templates_df = pd.read_csv("data/templates.csv")
    
    # Prepare data mappings
    processor = AttributeDataProcessor(config)
    template_mappings, attribute_mappings = processor.prepare_data(
        attributes_df, templates_df
    )
    
    # Create datasets
    train_dataset = ContrastiveDataset(
        attribute_mappings, template_mappings, config, "train"
    )
    val_dataset = ContrastiveDataset(
        attribute_mappings, template_mappings, config, "val"
    )
    
    #
