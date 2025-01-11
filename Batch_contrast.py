import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import argparse
import yaml
import json
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import multiprocessing
from functools import partial
import math

# Enhanced logging setup with file handler
def setup_logging(log_dir: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Optimized data structures
@dataclass
class PredictionResult:
    domain: str
    concept: str
    confidence: float
    processing_time: float  # Added for performance monitoring

@dataclass
class SamplingResult:
    positive_pairs: List[Tuple[str, str]]
    negative_pairs: List[Tuple[str, str, float]]
    stats: Dict[str, Any]  # Enhanced stats

@dataclass
class TrainingBatch:
    anchors: List[str]
    positives: List[str]
    negatives: List[str]
    negative_weights: List[float]
    
    def to_device(self, device: torch.device) -> 'TrainingBatch':
        """Move batch data to specified device."""
        return TrainingBatch(
            anchors=self.anchors,
            positives=self.positives,
            negatives=self.negatives,
            negative_weights=torch.tensor(self.negative_weights, device=device)
        )

@dataclass
class TrainingStats:
    epoch: int
    loss: float
    val_loss: float
    num_samples: int
    positive_pairs: int
    hard_negatives: int
    medium_negatives: int
    learning_rate: float
    batch_size: int
    epoch_time: float  # Added for performance monitoring
    throughput: float  # Added for performance monitoring
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'num_samples': self.num_samples,
            'positive_pairs': self.positive_pairs,
            'hard_negatives': self.hard_negatives,
            'medium_negatives': self.medium_negatives,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch_time': self.epoch_time,
            'throughput': self.throughput,
            'timestamp': datetime.now().isoformat()
        }

class OptimizedDataProcessor:
    def __init__(self, 
                attributes_path: str, 
                concepts_path: str, 
                test_size: float = 0.2,
                random_state: int = 42,
                num_workers: int = None):
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.test_size = test_size
        self.random_state = random_state
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self.attributes_df = None
        self.concepts_df = None
        self.domain_concepts = {}
        self.concept_embeddings_cache = {}
        
        self.train_indices = None
        self.val_indices = None
        
        self.load_data()

    def load_data(self) -> None:
        """Load data using parallel processing for large files."""
        try:
            self.logger.info("Loading data in parallel...")
            
            # Load CSVs in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                attr_future = executor.submit(pd.read_csv, self.attributes_path)
                concepts_future = executor.submit(pd.read_csv, self.concepts_path)
                
                self.attributes_df = attr_future.result()
                self.concepts_df = concepts_future.result()
            
            # Validate columns
            self._validate_columns()
            
            # Create stratified split
            self._create_split()
            
            # Build caches in parallel
            self._build_caches()
            
            self.logger.info(
                f"Loaded {len(self.attributes_df)} attributes and "
                f"{len(self.concepts_df)} concepts with {len(self.domain_concepts)} domains"
            )
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_columns(self) -> None:
        """Validate required columns exist."""
        required_attr_cols = ['attribute_name', 'description', 'domain', 'concept']
        required_concept_cols = ['domain', 'concept', 'concept_definition']
        
        missing_attr = [col for col in required_attr_cols if col not in self.attributes_df.columns]
        missing_concept = [col for col in required_concept_cols if col not in self.concepts_df.columns]
        
        if missing_attr or missing_concept:
            raise ValueError(
                f"Missing columns - Attributes: {missing_attr}, Concepts: {missing_concept}"
            )

    def _create_split(self) -> None:
        """Create stratified train/validation split."""
        self.train_indices, self.val_indices = train_test_split(
            np.arange(len(self.attributes_df)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.attributes_df['domain'].values
        )

    def _build_caches(self) -> None:
        """Build domain-concept cache in parallel."""
        def process_concept(row):
            domain, concept = row['domain'], row['concept']
            return domain, concept
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(process_concept, row) 
                for _, row in self.concepts_df.iterrows()
            ]
            
            for future in futures:
                domain, concept = future.result()
                if domain not in self.domain_concepts:
                    self.domain_concepts[domain] = set()
                self.domain_concepts[domain].add(concept)

class OptimizedPairSampler:
    def __init__(self, 
                data_processor: OptimizedDataProcessor,
                batch_size: int = 32,
                cache_size: int = 10000):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)
        self.similarity_cache = {}
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def compute_description_similarity(self, desc1: str, desc2: str) -> float:
        """Compute similarity using sentence transformers directly."""
        # Encode descriptions
        embeddings1 = self.model.encode(desc1, convert_to_tensor=True)
        embeddings2 = self.model.encode(desc2, convert_to_tensor=True)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=0)
        embeddings2 = F.normalize(embeddings2, p=2, dim=0)
        
        # Compute cosine similarity
        similarity = torch.dot(embeddings1, embeddings2).item()
        
        return similarity

    def _sample_hard_negatives(self, domain: str, concept: str, 
                             positive_attrs: pd.DataFrame, n_required: int,
                             attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample hard negatives based on description similarity."""
        hard_negatives = []
        
        # Get all potential negatives
        other_mask = attributes_df['domain'] != domain | \
                     (attributes_df['domain'] == domain & attributes_df['concept'] != concept)
        potential_negatives = attributes_df[other_mask]
        
        # Calculate similarities
        all_similarities = []
        all_rows = []
        
        # Compute similarities between each positive and potential negative
        with ThreadPoolExecutor() as executor:
            futures = []
            for _, pos_row in positive_attrs.iterrows():
                pos_desc = pos_row['description']
                for _, neg_row in potential_negatives.iterrows():
                    futures.append(
                        executor.submit(
                            self.compute_description_similarity,
                            pos_desc,
                            neg_row['description']
                        )
                    )
                    all_rows.append(neg_row)
                    
            # Collect similarities
            similarities = [f.result() for f in futures]
            all_similarities.extend(similarities)
            
        # Create array for easier filtering    
        similarities_array = np.array(all_similarities)
        high_sim_indices = np.where(similarities_array > 0.8)[0]
        
        # Sort by similarity to get hardest negatives first
        sorted_indices = high_sim_indices[np.argsort(-similarities_array[high_sim_indices])]
        
        # Take top n_required high similarity negatives
        for idx in sorted_indices[:n_required]:
            row = all_rows[idx]
            neg_text = self.data_processor.get_attribute_text(row)
            neg_def = self.data_processor.get_concept_definition(row['domain'], row['concept'])
            
            # Weight based on domain
            weight = 1.5 if row['domain'] == domain else 1.3
            hard_negatives.append((neg_text, neg_def, weight))
            
        return hard_negatives
        
    def compute_similarities_batch(self, 
                                descriptions1: List[str],
                                descriptions2: List[str],
                                batch_size: int = 512) -> np.ndarray:
        """Compute similarities for batches of descriptions."""
        similarities = []
        
        for i in range(0, len(descriptions1), batch_size):
            batch1 = descriptions1[i:i + batch_size]
            
            for j in range(0, len(descriptions2), batch_size):
                batch2 = descriptions2[j:j + batch_size]
                
                # Create cache key
                cache_key = (tuple(batch1), tuple(batch2))
                
                if cache_key in self.similarity_cache:
                    batch_similarities = self.similarity_cache[cache_key]
                else:
                    embeddings = util.semantic_search(
                        " ".join(batch1),
                        [" ".join(batch2)],
                        top_k=1
                    )[0]
                    batch_similarities = util.normalize_embeddings(embeddings)
                    
                    # Cache result
                    if len(self.similarity_cache) < self.cache_size:
                        self.similarity_cache[cache_key] = batch_similarities
                
                similarities.extend(batch_similarities)
        
        return np.array([s['score'] for s in similarities])

    def sample_pairs_batch(self,
                        domain: str,
                        concept: str,
                        attributes_df: pd.DataFrame,
                        batch_size: int = 512) -> SamplingResult:
        """Sample pairs in batches for better performance."""
        try:
            # Get positive attributes
            mask = (attributes_df['domain'] == domain) & \
                  (attributes_df['concept'] == concept)
            positive_attrs = attributes_df[mask]
            
            if len(positive_attrs) == 0:
                raise ValueError(f"No attributes found for {domain}-{concept}")
            
            concept_def = self.data_processor.get_concept_definition(domain, concept)
            
            # Create positive pairs in parallel
            with ThreadPoolExecutor(max_workers=self.data_processor.num_workers) as executor:
                positive_pairs = list(executor.map(
                    lambda row: (
                        self.data_processor.get_attribute_text(row),
                        concept_def
                    ),
                    [row for _, row in positive_attrs.iterrows()]
                ))
            
            # Sample negatives in batches
            n_required = len(positive_pairs)
            n_hard = int(0.4 * n_required)
            
            hard_negatives = self._sample_hard_negatives_batch(
                domain, concept, positive_attrs, n_hard, attributes_df, batch_size
            )
            
            n_medium = n_required - len(hard_negatives)
            medium_negatives = self._sample_medium_negatives_batch(
                domain, concept, n_medium, attributes_df, batch_size
            )
            
            negative_pairs = hard_negatives + medium_negatives
            
            stats = {
                'n_positives': len(positive_pairs),
                'n_hard_negatives': len(hard_negatives),
                'n_medium_negatives': len(medium_negatives),
                'sampling_time': time.time() - start_time
            }
            
            return SamplingResult(positive_pairs, negative_pairs, stats)
            
        except Exception as e:
            self.logger.error(f"Error sampling pairs: {str(e)}")
            raise

class OptimizedModelTrainer:
    def __init__(self,
                model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                batch_size: int = 32,
                num_epochs: int = 10,
                learning_rate: float = 2e-5,
                temperature: float = 0.07,
                output_dir: str = './model_output',
                patience: int = 5,
                num_workers: int = None,
                mixed_precision: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.output_dir = output_dir
        self.patience = patience
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.mixed_precision = mixed_precision
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and device
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if mixed_precision else None

    def train(self, data_processor: OptimizedDataProcessor, 
             sampler: OptimizedPairSampler) -> None:
        """Train model with optimizations."""
        try:
            self.logger.info("Preparing training data...")
            start_time = time.time()
            
            # Initialize training components
            metrics_tracker = MetricsTracker(self.output_dir)
            early_stopping = EarlyStopping(patience=self.patience)
            
            # Prepare batches in parallel
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                train_futures = [
                    executor.submit(
                        self.prepare_batches,
                        data_processor,
                        sampler,
                        True,
                        batch_idx
                    )
                    for batch_idx in range(0, len(data_processor.train_indices), self.batch_size)
                ]
                
                val_futures = [
                    executor.submit(
                        self.prepare_batches,
                        data_processor,
                        sampler,
                        False,
                        batch_idx
                    )
                    for batch_idx in range(0, len(data_processor.val_indices), self.batch_size)
                ]
                
                train_batches = [f.result() for f in train_futures]
                val_batches = [f.result() for f in val_futures]
            
            train_dataset = OptimizedConceptDataset(train_batches)
            val_dataset = OptimizedConceptDataset(val_batches)
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            # Initialize loss and optimizer
            loss_fn = OptimizedInfoNCELoss(
                self.model,
                self.temperature,
                self.device
            )
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            
            # Add learning rate scheduler
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,  # Restart every 5 epochs
                T_mult=2,  # Double the restart interval after each restart
                eta_min=1e-6  # Minimum learning rate
            )
            
            self.logger.info(
                f"Starting training for {self.num_epochs} epochs with "
                f"batch size {self.batch_size} on device {self.device}"
            )
            
            for epoch in range(self.num_epochs):
                epoch_start = time.time()
                
                # Training phase
                train_loss = self._train_epoch(
                    train_dataloader,
                    loss_fn,
                    optimizer,
                    scheduler
                )
                
                # Validation phase
                val_loss = self._validate(val_dataloader, loss_fn)
                
                epoch_time = time.time() - epoch_start
                throughput = len(train_dataset) / epoch_time
                
                # Update metrics
                metrics_tracker.add_metric('train_loss', train_loss, epoch)
                metrics_tracker.add_metric('val_loss', val_loss, epoch)
                metrics_tracker.add_metric('learning_rate', scheduler.get_last_lr()[0], epoch)
                metrics_tracker.add_metric('epoch_time', epoch_time, epoch)
                metrics_tracker.add_metric('throughput', throughput, epoch)
                
                # Create training stats
                stats = TrainingStats(
                    epoch=epoch,
                    loss=train_loss,
                    val_loss=val_loss,
                    num_samples=len(train_dataset),
                    positive_pairs=sum(len(b.positives) for b in train_batches),
                    hard_negatives=sum(sum(1 for w in b.negative_weights if w > 1.0) 
                                     for b in train_batches),
                    medium_negatives=sum(sum(1 for w in b.negative_weights if w == 1.0) 
                                       for b in train_batches),
                    learning_rate=scheduler.get_last_lr()[0],
                    batch_size=self.batch_size,
                    epoch_time=epoch_time,
                    throughput=throughput
                )
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}, "
                    f"Time: {epoch_time:.2f}s, "
                    f"Throughput: {throughput:.2f} samples/s"
                )
                
                # Early stopping check
                if early_stopping(val_loss, self.model, epoch, self.output_dir):
                    self.logger.info(
                        f"Early stopping triggered after {epoch+1} epochs"
                    )
                    break
                
                # Checkpoint saving
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch)
                
                # Update scheduler
                scheduler.step()
            
            # Save final metrics and model
            metrics_tracker.save_metrics()
            self._save_final_model(early_stopping)
            
            training_time = time.time() - start_time
            self.logger.info(
                f"Training completed in {training_time:.2f}s "
                f"({training_time/3600:.2f}h)"
            )
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

    def _train_epoch(self, 
                    dataloader: DataLoader,
                    loss_fn: 'OptimizedInfoNCELoss',
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler) -> float:
        """Train for one epoch with optimizations."""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                batch = batch.to_device(self.device)
                
                optimizer.zero_grad()
                
                if self.mixed_precision:
                    with autocast():
                        loss = loss_fn(
                            batch.anchors,
                            batch.positives,
                            batch.negatives,
                            batch.negative_weights
                        )
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss = loss_fn(
                        batch.anchors,
                        batch.positives,
                        batch.negatives,
                        batch.negative_weights
                    )
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / batch_count,
                    'lr': scheduler.get_last_lr()[0]
                })
        
        return total_loss / batch_count

    def _validate(self, 
                 dataloader: DataLoader,
                 loss_fn: 'OptimizedInfoNCELoss') -> float:
        """Run validation with optimizations."""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as pbar:
                for batch in pbar:
                    batch = batch.to_device(self.device)
                    
                    if self.mixed_precision:
                        with autocast():
                            loss = loss_fn(
                                batch.anchors,
                                batch.positives,
                                batch.negatives,
                                batch.negative_weights
                            )
                    else:
                        loss = loss_fn(
                            batch.anchors,
                            batch.positives,
                            batch.negatives,
                            batch.negative_weights
                        )
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    pbar.set_postfix({'val_loss': total_loss / batch_count})
        
        return total_loss / batch_count

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
        self.model.save(save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")

    def _save_final_model(self, early_stopping: EarlyStopping) -> None:
        """Save final model."""
        if early_stopping.best_model_path:
            self.model = SentenceTransformer(early_stopping.best_model_path)
            self.logger.info(
                f"Loaded best model from {early_stopping.best_model_path}"
            )
        else:
            final_path = os.path.join(self.output_dir, "final-model")
            self.model.save(final_path)
            self.logger.info(f"Saved final model to {final_path}")

class OptimizedInfoNCELoss(nn.Module):
    def __init__(self, 
                model: SentenceTransformer,
                temperature: float = 0.07,
                device: torch.device = None):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self,
               anchors: List[str],
               positives: List[str],
               negatives: List[str],
               negative_weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted InfoNCE loss with optimized encoding."""
        # Encode texts in batches
        def encode_batch(texts: List[str], batch_size: int = 512) -> torch.Tensor:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                )
                embeddings.append(batch_embeddings)
            return torch.cat(embeddings)
        
        # Encode all texts in parallel using multiple GPUs if available
        anchor_embeddings = encode_batch(anchors)
        positive_embeddings = encode_batch(positives)
        negative_embeddings = encode_batch(negatives)
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        
        # Compute similarities efficiently using matrix multiplication
        positive_similarities = torch.sum(
            anchor_embeddings * positive_embeddings,
            dim=-1
        ) / self.temperature
        
        negative_similarities = torch.matmul(
            anchor_embeddings,
            negative_embeddings.transpose(0, 1)
        ) / self.temperature
        
        # Apply weights to negative similarities
        negative_similarities = negative_similarities * negative_weights.unsqueeze(0)
        
        # Compute InfoNCE loss
        logits = torch.cat([
            positive_similarities.unsqueeze(-1),
            negative_similarities
        ], dim=-1)
        
        labels = torch.zeros(
            len(anchors),
            dtype=torch.long,
            device=logits.device
        )
        
        return F.cross_entropy(logits, labels)

class OptimizedPredictor:
    def __init__(self,
                model_path: str,
                data_processor: OptimizedDataProcessor,
                batch_size: int = 32,
                top_k: int = 3,
                num_workers: int = None,
                cache_size: int = 10000):
        self.model_path = model_path
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.top_k = top_k
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.cache_size = cache_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and device
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize caches
        self.embedding_cache = {}
        self.prediction_cache = {}
        
    def _load_model(self) -> SentenceTransformer:
        """Load model with error handling."""
        try:
            return SentenceTransformer(self.model_path)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_batch(self,
                    texts: List[Tuple[str, str]],
                    batch_size: int = 512) -> List[List[PredictionResult]]:
        """Predict in batches with caching."""
        try:
            start_time = time.time()
            
            # Combine attribute texts
            combined_texts = [
                f"{name} {desc}" for name, desc in texts
            ]
            
            # Get text embeddings in batches
            text_embeddings = []
            for i in range(0, len(combined_texts), batch_size):
                batch = combined_texts[i:i + batch_size]
                
                # Check cache first
                cache_key = tuple(batch)
                if cache_key in self.embedding_cache:
                    batch_embeddings = self.embedding_cache[cache_key]
                else:
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                    
                    # Cache embeddings
                    if len(self.embedding_cache) < self.cache_size:
                        self.embedding_cache[cache_key] = batch_embeddings
                
                text_embeddings.append(batch_embeddings)
            
            text_embeddings = torch.cat(text_embeddings)
            
            # Get concept embeddings (cached)
            concept_embeddings = self._get_concept_embeddings()
            
            # Compute similarities efficiently
            similarities = torch.matmul(
                text_embeddings,
                concept_embeddings.transpose(0, 1)
            )
            
            # Get top-k predictions for each text
            top_k_values, top_k_indices = similarities.topk(self.top_k)
            
            # Convert to prediction results
            results = []
            for i in range(len(texts)):
                text_results = []
                for j in range(self.top_k):
                    concept_idx = top_k_indices[i][j].item()
                    confidence = top_k_values[i][j].item()
                    
                    domain, concept = self.data_processor.get_domain_concept(concept_idx)
                    text_results.append(
                        PredictionResult(
                            domain=domain,
                            concept=concept,
                            confidence=confidence,
                            processing_time=(time.time() - start_time) / len(texts)
                        )
                    )
                results.append(text_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise

    def _get_concept_embeddings(self) -> torch.Tensor:
        """Get or compute concept embeddings with caching."""
        cache_key = 'concept_embeddings'
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        concept_texts = []
        for domain in self.data_processor.get_domains():
            for concept in self.data_processor.get_concepts_for_domain(domain):
                definition = self.data_processor.get_concept_definition(domain, concept)
                concept_texts.append(definition)
        
        embeddings = self.model.encode(
            concept_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=self.batch_size
        )
        
        self.embedding_cache[cache_key] = embeddings
        return embeddings

    def predict_csv(self, input_path: str, output_path: str) -> None:
        """Predict for CSV file with batched processing."""
        try:
            self.logger.info(f"Reading input CSV from {input_path}")
            df = pd.read_csv(input_path)
            
            required_cols = ['attribute_name', 'description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Process in batches
            all_predictions = []
            texts = list(zip(df['attribute_name'], df['description']))
            
            with tqdm(total=len(texts), desc="Predicting") as pbar:
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i + self.batch_size]
                    batch_predictions = self.predict_batch(batch_texts)
                    all_predictions.extend(batch_predictions)
                    pbar.update(len(batch_texts))
            
            # Convert predictions to DataFrame format
            result_rows = []
            for i, predictions in enumerate(all_predictions):
                top_pred = predictions[0]  # Take top prediction
                result_rows.append({
                    'attribute_name': df.iloc[i]['attribute_name'],
                    'description': df.iloc[i]['description'],
                    'predicted_domain': top_pred.domain,
                    'predicted_concept': top_pred.concept,
                    'confidence': top_pred.confidence,
                    'processing_time': top_pred.processing_time
                })
            
            result_df = pd.DataFrame(result_rows)
            
            # Save results with performance metrics
            self.logger.info(f"Saving predictions to {output_path}")
            result_df.to_csv(output_path, index=False)
            
            # Log performance metrics
            avg_time = result_df['processing_time'].mean()
            throughput = 1.0 / avg_time if avg_time > 0 else 0
            
            self.logger.info(
                f"Prediction complete! "
                f"Average processing time: {avg_time:.3f}s per sample, "
                f"Throughput: {throughput:.2f} samples/s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise

def predict_interactive(predictor: OptimizedPredictor):
    """Interactive prediction mode with enhanced feedback."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive prediction mode. Type 'quit' to exit.")
    
    try:
        while True:
            attribute_name = input("\nEnter attribute name (or 'quit' to exit): ").strip()
            if attribute_name.lower() == 'quit':
                break
            
            description = input("Enter description: ").strip()
            
            start_time = time.time()
            predictions = predictor.predict_batch([(attribute_name, description)])[0]
            proc_time = time.time() - start_time
            
            print("\nPredictions:")
            print("-" * 50)
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. Domain: {pred.domain}")
                print(f"   Concept: {pred.concept}")
                print(f"   Confidence: {pred.confidence:.4f}")
                print("-" * 50)
            
            print(f"\nProcessing time: {proc_time:.3f}s")
            
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        print("An error occurred. Please try again.")

def setup_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """Create and setup experiment directory with enhanced organization."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    experiment_dir = Path(base_dir) / dir_name
    
    # Create required subdirectories with gitkeep files
    subdirs = ['models', 'logs', 'metrics', 'predictions', 'checkpoints', 'analysis']
    for subdir in subdirs:
        subdir_path = experiment_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        (subdir_path / '.gitkeep').touch()
    
    # Create experiment info file
    info = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'subdirectories': subdirs,
        'platform': sys.platform,
        'python_version': sys.version,
        'cuda_available': torch.cuda.is_available()
    }
    
    with open(experiment_dir / 'experiment_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return experiment_dir

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration with enhanced checks."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Required configuration sections and their required fields
        required_config = {
            'data': ['attributes_path', 'concepts_path'],
            'model': ['name', 'batch_size', 'temperature'],
            'training': ['num_epochs', 'learning_rate', 'patience'],
            'prediction': ['batch_size', 'top_k'],
            'output_dir': None
        }
        
        # Validate sections and fields
        for section, fields in required_config.items():
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
            
            if fields:
                for field in fields:
                    if field not in config[section]:
                        raise ValueError(f"Missing required field '{field}' in section '{section}'")
        
        # Validate data paths
        data_paths = [
            Path(config['data']['attributes_path']),
            Path(config['data']['concepts_path'])
        ]
        
        for path in data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # Validate numeric parameters
        numeric_params = [
            (config['model']['batch_size'], 'model.batch_size', 1),
            (config['model']['temperature'], 'model.temperature', 0),
            (config['training']['num_epochs'], 'training.num_epochs', 1),
            (config['training']['learning_rate'], 'training.learning_rate', 0),
            (config['prediction']['batch_size'], 'prediction.batch_size', 1),
            (config['prediction']['top_k'], 'prediction.top_k', 1)
        ]
        
        for value, name, min_value in numeric_params:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {name} must be numeric")
            if value < min_value:
                raise ValueError(f"Parameter {name} must be >= {min_value}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error in configuration: {str(e)}")
        raise

class EarlyStopping:
    """Early stopping handler with optimization for checkpointing."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_path = None
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int, save_path: str) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path, epoch)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, save_path, epoch)
            self.counter = 0
            
        return self.early_stop
        
    def save_checkpoint(self, model: nn.Module, save_path: str, epoch: int):
        """Save model checkpoint with optimized I/O."""
        path = f"{save_path}/best_model_epoch_{epoch}"
        model.save(path)
        
        # Remove previous best model to save space
        if self.best_model_path and os.path.exists(self.best_model_path):
            try:
                shutil.rmtree(self.best_model_path)
            except Exception as e:
                logging.warning(f"Could not remove old model: {e}")
        
        self.best_model_path = path

class MetricsTracker:
    """Enhanced metrics tracking with performance monitoring."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: Dict[str, float] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        
    def add_metric(self, name: str, value: float, epoch: Optional[int] = None) -> None:
        """Add metric with timestamp."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        if epoch is not None:
            self.epoch_metrics[f"{name}_epoch_{epoch}"] = value
        
        # Add timestamp for performance tracking
        timestamp = datetime.now().timestamp()
        if name not in self.performance_metrics:
            self.performance_metrics[name] = []
        self.performance_metrics[name].append((timestamp, value))
            
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save metrics with enhanced error handling and backup."""
        try:
            output_path = self.output_dir / filename
            backup_path = self.output_dir / f"{filename}.backup"
            
            metrics_data = {
                'metrics': self.metrics,
                'epoch_metrics': self.epoch_metrics,
                'performance_metrics': self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to backup first
            with open(backup_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Then rename to actual file
            if os.path.exists(output_path):
                os.replace(str(backup_path), str(output_path))
            else:
                os.rename(str(backup_path), str(output_path))
                
            logging.info(f"Saved metrics to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")
            if os.path.exists(backup_path):
                logging.info(f"Backup metrics saved at {backup_path}")

class OptimizedConceptDataset(Dataset):
    """Optimized dataset implementation with caching."""
    def __init__(self, batches: List[TrainingBatch]):
        self.batches = batches
        self._length = len(batches)
        
    def __len__(self) -> int:
        return self._length
        
    def __getitem__(self, idx: int) -> TrainingBatch:
        if not 0 <= idx < self._length:
            raise IndexError("Dataset index out of range")
        return self.batches[idx]
    
    def get_batch_sizes(self) -> List[int]:
        """Get sizes of all batches for optimization."""
        return [len(batch.anchors) for batch in self.batches]
    
    def get_total_samples(self) -> int:
        """Get total number of samples across all batches."""
        return sum(self.get_batch_sizes())

def main():
    """Main execution flow with enhanced error handling and logging."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Train or predict with optimized sentence transformer model'
        )
        parser.add_argument('--config', type=str, help='Path to config YAML file')
        parser.add_argument(
            '--mode',
            choices=['train', 'predict', 'interactive'],
            required=True,
            help='Operation mode: train, predict, or interactive'
        )
        parser.add_argument(
            '--attributes',
            type=str,
            required=True,
            help='Path to attributes CSV file'
        )
        parser.add_argument(
            '--concepts',
            type=str,
            required=True,
            help='Path to concepts CSV file'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='./output',
            help='Output directory for model and predictions'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            help='Path to saved model (required for predict mode)'
        )
        parser.add_argument(
            '--input',
            type=str,
            help='Input CSV file for prediction'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training and prediction'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=10,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Fraction of data to use for validation'
        )
        parser.add_argument(
            '--random-state',
            type=int,
            default=42,
            help='Random seed for reproducibility'
        )
        parser.add_argument(
            '--num-workers',
            type=int,
            default=None,
            help='Number of worker processes (default: CPU count - 1)'
        )
        parser.add_argument(
            '--mixed-precision',
            action='store_true',
            help='Enable mixed precision training'
        )
        
        args = parser.parse_args()
        
        # Setup experiment directory and logging
        experiment_dir = setup_experiment_dir(
            args.output_dir,
            f"{args.mode}_experiment"
        )
        logger = setup_logging(experiment_dir / 'logs')
        
        # Log system information
        logger.info("System Information:")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = OptimizedDataProcessor(
            args.attributes,
            args.concepts,
            test_size=args.test_size,
            random_state=args.random_state,
            num_workers=args.num_workers
        )
        
        if args.mode == 'train':
            # Training mode
            logger.info("Initializing training components...")
            sampler = OptimizedPairSampler(data_processor, args.batch_size)
            trainer = OptimizedModelTrainer(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                output_dir=experiment_dir / 'models',
                num_workers=args.num_workers,
                mixed_precision=args.mixed_precision
            )
            
            trainer.train(data_processor, sampler)
            
        elif args.mode == 'predict':
            # Prediction mode
            if not args.model_path:
                raise ValueError("--model-path required for predict mode")
            if not args.input:
                raise ValueError("--input required for predict mode")
            
            logger.info("Initializing predictor...")
            predictor = OptimizedPredictor(
                model_path=args.model_path,
                data_processor=data_processor,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            output_path = experiment_dir / 'predictions' / 'predictions.csv'
            predictor.predict_csv(args.input, output_path)
            
        else:  # interactive mode
            if not args.model_path:
                raise ValueError("--model-path required for interactive mode")
            
            logger.info("Initializing predictor for interactive mode...")
            predictor = OptimizedPredictor(
                model_path=args.model_path,
                data_processor=data_processor,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            predict_interactive(predictor)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
