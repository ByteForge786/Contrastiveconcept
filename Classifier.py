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
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import yaml
import json
import sys
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Dataclasses for structured data handling
@dataclass
class PredictionResult:
    domain: str
    concept: str
    confidence: float

@dataclass
class SamplingResult:
    positive_pairs: List[Tuple[str, str]]  # (attribute_text, concept_definition)
    negative_pairs: List[Tuple[str, str, float]]  # (attribute_text, concept_definition, weight)
    stats: Dict[str, int]

@dataclass
class TrainingBatch:
    anchors: List[str]  # concept definitions
    positives: List[str]  # positive attribute texts
    negatives: List[str]  # negative attribute texts
    negative_weights: List[float]  # weights for negative samples

@dataclass
class TrainingStats:
    """Container for training statistics."""
    epoch: int
    loss: float
    num_samples: int
    positive_pairs: int
    hard_negatives: int
    medium_negatives: int
    learning_rate: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'num_samples': self.num_samples,
            'positive_pairs': self.positive_pairs,
            'hard_negatives': self.hard_negatives,
            'medium_negatives': self.medium_negatives,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'timestamp': datetime.now().isoformat()
        }

class MetricsTracker:
    """Track and log training/prediction metrics."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: Dict[str, float] = {}
        
    def add_metric(self, name: str, value: float, epoch: Optional[int] = None) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        if epoch is not None:
            self.epoch_metrics[f"{name}_epoch_{epoch}"] = value
            
    def save_metrics(self, filename: str = "metrics.json") -> None:
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': self.metrics,
                    'epoch_metrics': self.epoch_metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

class ConceptDataset(Dataset):
    """Dataset class for training data."""
    def __init__(self, sampling_results: List[TrainingBatch]):
        self.batches = sampling_results
        
    def __len__(self) -> int:
        return len(self.batches)
        
    def __getitem__(self, idx: int) -> TrainingBatch:
        return self.batches[idx]

  class DataProcessor:
    def __init__(self, attributes_path: str, concepts_path: str):
        """Initialize DataProcessor with paths to CSV files."""
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.logger = logging.getLogger(__name__)
        
        # Load and validate data
        self.attributes_df = None
        self.concepts_df = None
        self.domain_concepts = {}  # Cache for domain-concept pairs
        self.load_data()

    def load_data(self) -> None:
        """Load and validate CSV files."""
        try:
            self.logger.info("Loading attributes data...")
            self.attributes_df = pd.read_csv(self.attributes_path)
            required_attr_cols = ['attribute_name', 'description', 'domain', 'concept']
            self._validate_columns(self.attributes_df, required_attr_cols, 'attributes')

            self.logger.info("Loading concepts data...")
            self.concepts_df = pd.read_csv(self.concepts_path)
            required_concept_cols = ['domain', 'concept', 'concept_definition']
            self._validate_columns(self.concepts_df, required_concept_cols, 'concepts')

            # Build domain-concept cache
            self._build_domain_concepts_cache()
            
            self.logger.info(f"Loaded {len(self.attributes_df)} attributes and {len(self.concepts_df)} concept definitions")
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
        """Validate that DataFrame has required columns."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name} CSV: {missing_cols}")

    def _build_domain_concepts_cache(self) -> None:
        """Build cache of valid domain-concept pairs."""
        for _, row in self.concepts_df.iterrows():
            if row['domain'] not in self.domain_concepts:
                self.domain_concepts[row['domain']] = set()
            self.domain_concepts[row['domain']].add(row['concept'])
        
        self.logger.info(f"Found {len(self.domain_concepts)} domains with unique concepts")

    def get_attribute_text(self, row: pd.Series) -> str:
        """Combine attribute name and description."""
        return f"{row['attribute_name']} {row['description']}"

    def get_concept_definition(self, domain: str, concept: str) -> str:
        """Get concept definition for a domain-concept pair."""
        mask = (self.concepts_df['domain'] == domain) & (self.concepts_df['concept'] == concept)
        definitions = self.concepts_df[mask]['concept_definition'].values
        if len(definitions) == 0:
            raise ValueError(f"No definition found for domain={domain}, concept={concept}")
        return definitions[0]

    def get_domains(self) -> List[str]:
        """Get list of unique domains."""
        return list(self.domain_concepts.keys())

    def get_concepts_for_domain(self, domain: str) -> List[str]:
        """Get list of concepts for a given domain."""
        return list(self.domain_concepts.get(domain, set()))

class PairSampler:
    def __init__(self, data_processor: DataProcessor, batch_size: int = 32):
        """Initialize sampler with DataProcessor instance."""
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def compute_description_similarity(self, desc1: str, desc2: str) -> float:
        """Compute cosine similarity between two descriptions."""
        embeddings = util.normalize_embeddings(
            util.semantic_search(desc1, [desc2], top_k=1)[0]
        )
        return float(embeddings[0]['score'])

  def sample_pairs(self, domain: str, concept: str) -> SamplingResult:
        """Sample positive and negative pairs for a domain-concept pair."""
        try:
            # Get all positive attributes
            mask = (self.data_processor.attributes_df['domain'] == domain) & \
                  (self.data_processor.attributes_df['concept'] == concept)
            positive_attrs = self.data_processor.attributes_df[mask]
            
            if len(positive_attrs) == 0:
                raise ValueError(f"No attributes found for domain={domain}, concept={concept}")

            concept_def = self.data_processor.get_concept_definition(domain, concept)
            
            # Create positive pairs
            positive_pairs = [
                (self.data_processor.get_attribute_text(row), concept_def)
                for _, row in positive_attrs.iterrows()
            ]
            
            n_required_negatives = len(positive_pairs)
            
            # Sample hard negatives (40%)
            n_hard = int(0.4 * n_required_negatives)
            hard_negatives = self._sample_hard_negatives(domain, concept, positive_attrs, n_hard)
            
            # Sample medium negatives (60% or remainder)
            n_medium = n_required_negatives - len(hard_negatives)
            medium_negatives = self._sample_medium_negatives(domain, concept, n_medium)
            
            # Combine all negatives
            negative_pairs = hard_negatives + medium_negatives
            
            stats = {
                'n_positives': len(positive_pairs),
                'n_hard_negatives': len(hard_negatives),
                'n_medium_negatives': len(medium_negatives)
            }
            
            self.logger.info(
                f"Sampled pairs for {domain}-{concept}: "
                f"{stats['n_positives']} positives, "
                f"{stats['n_hard_negatives']} hard negatives, "
                f"{stats['n_medium_negatives']} medium negatives"
            )
            
            return SamplingResult(positive_pairs, negative_pairs, stats)
            
        except Exception as e:
            self.logger.error(f"Error sampling pairs for {domain}-{concept}: {str(e)}")
            raise

    def _sample_hard_negatives(self, domain: str, concept: str, 
                             positive_attrs: pd.DataFrame, n_required: int) -> List[Tuple[str, str, float]]:
        """Sample hard negatives from same domain but different concepts."""
        hard_negatives = []
        other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                         if c != concept]
        
        if not other_concepts:
            self.logger.warning(f"No other concepts found for domain {domain}")
            return hard_negatives
        
        # Get all potential hard negative attributes
        mask = (self.data_processor.attributes_df['domain'] == domain) & \
               (self.data_processor.attributes_df['concept'].isin(other_concepts))
        potential_negatives = self.data_processor.attributes_df[mask]
        
        if len(potential_negatives) == 0:
            self.logger.warning(f"No potential hard negatives found for {domain}-{concept}")
            return hard_negatives
            
        # Compute similarities in parallel
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
            
            similarities = [f.result() for f in futures]
        
        # Filter for high similarity (>80%) and add with weight 1.5
        high_sim_mask = np.array(similarities) > 0.8
        high_sim_negatives = potential_negatives.iloc[high_sim_mask]
        
        for _, row in high_sim_negatives.iterrows():
            neg_text = self.data_processor.get_attribute_text(row)
            neg_def = self.data_processor.get_concept_definition(domain, row['concept'])
            hard_negatives.append((neg_text, neg_def, 1.5))
            
            if len(hard_negatives) >= n_required:
                break
                
        return hard_negatives

    def _sample_medium_negatives(self, domain: str, concept: str, 
                               n_required: int) -> List[Tuple[str, str, float]]:
        """Sample medium negatives from different domains."""
        medium_negatives = []
        other_domains = [d for d in self.data_processor.get_domains() if d != domain]
        
        if not other_domains:
            self.logger.warning("No other domains found for medium negatives")
            return medium_negatives
            
        sample_domains = np.random.choice(other_domains, n_required, replace=True)
        
        for d in sample_domains:
            concepts = self.data_processor.get_concepts_for_domain(d)
            if not concepts:
                continue
                
            c = np.random.choice(list(concepts))
            
            mask = (self.data_processor.attributes_df['domain'] == d) & \
                   (self.data_processor.attributes_df['concept'] == c)
            neg_attrs = self.data_processor.attributes_df[mask]
            
            if len(neg_attrs) == 0:
                continue
                
            neg_row = neg_attrs.iloc[np.random.randint(len(neg_attrs))]
            neg_text = self.data_processor.get_attribute_text(neg_row)
            neg_def = self.data_processor.get_concept_definition(d, c)
            
            medium_negatives.append((neg_text, neg_def, 1.0))
            
            if len(medium_negatives) >= n_required:
                break
                
        return medium_negatives

class CustomInfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature: float = 0.07):
        super().__init__()
        self.model = model
        self.temperature = temperature
        
    def forward(self, anchors: List[str], positives: List[str], 
                negatives: List[str], negative_weights: List[float]) -> torch.Tensor:
        """Compute weighted InfoNCE loss with positive and negative pairs."""
        # Encode all texts
        anchor_embeddings = self.model.encode(anchors, convert_to_tensor=True)
        positive_embeddings = self.model.encode(positives, convert_to_tensor=True)
        negative_embeddings = self.model.encode(negatives, convert_to_tensor=True)
        
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        
        # Compute similarities
        positive_similarities = torch.sum(
            anchor_embeddings * positive_embeddings, dim=-1
        ) / self.temperature
        
        negative_similarities = torch.matmul(
            anchor_embeddings, negative_embeddings.transpose(0, 1)
        ) / self.temperature
        
        # Apply weights to negative similarities
        negative_weights = torch.tensor(negative_weights).to(negative_similarities.device)
        negative_similarities = negative_similarities * negative_weights.unsqueeze(0)
        
        # Compute InfoNCE loss
        logits = torch.cat([positive_similarities.unsqueeze(-1), negative_similarities], dim=-1)
        labels = torch.zeros(len(anchors), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

class ModelTrainer:
    def __init__(self, 
                model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                batch_size: int = 32,
                num_epochs: int = 10,
                learning_rate: float = 2e-5,
                temperature: float = 0.07,
                output_dir: str = './model_output'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def train(self, data_processor: DataProcessor, sampler: PairSampler) -> None:
        """Train the model using InfoNCE loss."""
        try:
            self.logger.info("Preparing training data...")
            
            # Initialize metrics tracker
            metrics_tracker = MetricsTracker(self.output_dir)
            
            # Prepare batches
            batches = []
            for domain in data_processor.get_domains():
                for concept in data_processor.get_concepts_for_domain(domain):
                    try:
                        # Sample pairs for this domain-concept
                        result = sampler.sample_pairs(domain, concept)
                        
                        # Create batch
                        batch = TrainingBatch(
                            anchors=[result.positive_pairs[0][1]] * len(result.positive_pairs),
                            positives=[pair[0] for pair in result.positive_pairs],
                            negatives=[pair[0] for pair in result.negative_pairs],
                            negative_weights=[pair[2] for pair in result.negative_pairs]
                        )
                        
                        batches.append(batch)
                        
                    except Exception as e:
                        self.logger.error(f"Error preparing batch for {domain}-{concept}: {str(e)}")
                        continue
            
            if not batches:
                raise ValueError("No training batches could be prepared")
                
            dataset = ConceptDataset(batches)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Initialize loss and optimizer
            loss_fn = CustomInfoNCELoss(self.model, self.temperature)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            self.logger.info(f"Starting training for {self.num_epochs} epochs...")
            
            for epoch in range(self.num_epochs):
                total_loss = 0
                batch_count = 0
                
                with tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
                    for batch in pbar:
                        optimizer.zero_grad()
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
                        pbar.set_postfix({'loss': total_loss / batch_count})
                
                avg_loss = total_loss / batch_count
                metrics_tracker.add_metric('loss', avg_loss, epoch)
                
                # Create training stats
                stats = TrainingStats(
                    epoch=epoch,
                    loss=avg_loss,
                    num_samples=len(dataset),
                    positive_pairs=sum(len(b.positives) for b in batches),
                    hard_negatives=sum(sum(1 for w in b.negative_weights if w > 1.0) for b in batches),
                    medium_negatives=sum(sum(1 for w in b.negative_weights if w == 1.0) for b in batches),
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size
                )
                
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {avg_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
                    self.model.save(save_path)
                    self.logger.info(f"Saved checkpoint to {save_path}")
            
            # Save final model and metrics
            final_path = os.path.join(self.output_dir, "final-model")
            self.model.save(final_path)
            metrics_tracker.save_metrics()
            self.logger.info(f"Saved final model to {final_path}")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

class ModelPredictor:
    def __init__(self, 
                model_path: str,
                data_processor: DataProcessor,
                batch_size: int = 32,
                top_k: int = 3):
        self.model_path = model_path
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.top_k = top_k
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self.model = SentenceTransformer(model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        # Cache concept definitions
        self.concept_definitions = self._cache_concept_definitions()
        
    def _cache_concept_definitions(self) -> Dict[Tuple[str, str], str]:
        """Cache all domain-concept definitions for faster prediction."""
        definitions = {}
        for domain in self.data_processor.get_domains():
            for concept in self.data_processor.get_concepts_for_domain(domain):
                try:
                    definition = self.data_processor.get_concept_definition(domain, concept)
                    definitions[(domain, concept)] = definition
                except Exception as e:
                    self.logger.warning(f"Could not cache definition for {domain}-{concept}: {str(e)}")
        return definitions
        
    def predict_single(self, attribute_name: str, description: str) -> List[PredictionResult]:
        """Predict domain-concept for a single attribute."""
        try:
            # Combine attribute text
            attribute_text = f"{attribute_name} {description}"
            
            # Encode attribute text
            attribute_embedding = self.model.encode(
                attribute_text,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Get similarities with all concept definitions
            results = []
            for (domain, concept), definition in self.concept_definitions.items():
                definition_embedding = self.model.encode(
                    definition,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Compute similarity
  similarity = torch.nn.functional.cosine_similarity(
                    attribute_embedding,
                    definition_embedding,
                    dim=0
                ).item()
                
                results.append(PredictionResult(domain, concept, similarity))
            
            # Sort by confidence and return top-k
            results.sort(key=lambda x: x.confidence, reverse=True)
            return results[:self.top_k]
            
        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            raise

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict domain-concepts for a batch of attributes."""
        try:
            # Validate input
            required_cols = ['attribute_name', 'description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            results = []
            
            # Process in batches
            for i in tqdm(range(0, len(df), self.batch_size)):
                batch_df = df.iloc[i:i+self.batch_size]
                
                # Process each row in batch
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for _, row in batch_df.iterrows():
                        futures.append(
                            executor.submit(
                                self.predict_single,
                                row['attribute_name'],
                                row['description']
                            )
                        )
                    
                    # Collect results
                    for future in futures:
                        predictions = future.result()
                        if predictions:  # Take top prediction
                            top_pred = predictions[0]
                            results.append({
                                'predicted_domain': top_pred.domain,
                                'predicted_concept': top_pred.concept,
                                'confidence': top_pred.confidence
                            })
                        else:
                            results.append({
                                'predicted_domain': None,
                                'predicted_concept': None,
                                'confidence': 0.0
                            })
            
            # Add predictions to DataFrame
            result_df = pd.DataFrame(results)
            return pd.concat([df.reset_index(drop=True), result_df], axis=1)
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise

    def predict_csv(self, input_path: str, output_path: str) -> None:
        """Predict domain-concepts for attributes in a CSV file."""
        try:
            # Read input CSV
            self.logger.info(f"Reading input CSV from {input_path}")
            df = pd.read_csv(input_path)
            
            # Make predictions
            self.logger.info("Making predictions...")
            result_df = self.predict_batch(df)
            
            # Save results
            self.logger.info(f"Saving predictions to {output_path}")
            result_df.to_csv(output_path, index=False)
            
            self.logger.info("Prediction complete!")
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise

def setup_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """Create and setup experiment directory with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
        
    experiment_dir = Path(base_dir) / dir_name
    
    # Create required subdirectories
    for subdir in ['models', 'logs', 'metrics', 'predictions']:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    return experiment_dir

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Required configuration sections
        required_sections = ['data', 'model', 'training', 'prediction', 'output_dir']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            raise ValueError(f"Missing required config sections: {missing_sections}")
            
        # Validate data paths exist
        data_paths = [
            Path(config['data']['attributes_path']),
            Path(config['data']['concepts_path'])
        ]
        
        for path in data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
                
        return config
        
    except Exception as e:
        logger.error(f"Error in configuration: {str(e)}")
        raise

def predict_interactive(predictor: ModelPredictor):
    """Interactive prediction mode."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive prediction mode. Type 'quit' to exit.")
    
    while True:
        try:
            attribute_name = input("\nEnter attribute name (or 'quit' to exit): ").strip()
            if attribute_name.lower() == 'quit':
                break
                
            description = input("Enter description: ").strip()
            
            predictions = predictor.predict_single(attribute_name, description)
            
            print("\nPredictions:")
            print("-" * 50)
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. Domain: {pred.domain}")
                print(f"   Concept: {pred.concept}")
                print(f"   Confidence: {pred.confidence:.4f}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            print("An error occurred. Please try again.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or predict with sentence transformer model')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive'], required=True,
                      help='Operation mode: train, predict, or interactive')
    parser.add_argument('--attributes', type=str, required=True,
                      help='Path to attributes CSV file')
    parser.add_argument('--concepts', type=str, required=True,
                      help='Path to concepts CSV file')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Output directory for model and predictions')
    parser.add_argument('--model-path', type=str,
                      help='Path to saved model (required for predict mode)')
    parser.add_argument('--input', type=str,
                      help='Input CSV file for prediction')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training and prediction')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    
    try:
        # Initialize data processor
        data_processor = DataProcessor(args.attributes, args.concepts)
        
        if args.mode == 'train':
            # Training mode
            sampler = PairSampler(data_processor, args.batch_size)
            trainer = ModelTrainer(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                output_dir=args.output_dir
            )
            
            trainer.train(data_processor, sampler)
            
        elif args.mode == 'predict':
            # Prediction mode
            if not args.model_path:
                raise ValueError("--model-path required for predict mode")
            if not args.input:
                raise ValueError("--input required for predict mode")
                
            predictor = ModelPredictor(
                model_path=args.model_path,
                data_processor=data_processor,
                batch_size=args.batch_size
            )
            
            output_path = os.path.join(args.output_dir, 'predictions.csv')
            predictor.predict_csv(args.input, output_path)
            
        else:  # interactive mode
            if not args.model_path:
                raise ValueError("--model-path required for interactive mode")
                
            predictor = ModelPredictor(
                model_path=args.model_path,
                data_processor=data_processor,
                batch_size=args.batch_size
            )
            
            predict_interactive(predictor)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
