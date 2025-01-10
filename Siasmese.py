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

@dataclass
class PredictionResult:
    domain: str
    concept: str
    confidence: float

@dataclass
class TrainingStats:
    """Container for training statistics."""
    epoch: int
    loss: float
    num_samples: int
    accuracy: float
    learning_rate: float
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'loss': self.loss,
            'num_samples': self.num_samples,
            'accuracy': self.accuracy,
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

class SiameseDataset(Dataset):
    """Dataset for Siamese Network training."""
    def __init__(self, 
                attributes_df: pd.DataFrame,
                concepts_df: pd.DataFrame,
                batch_size: int = 32):
        self.attributes_df = attributes_df
        self.concepts_df = concepts_df
        self.batch_size = batch_size
        
        # Create all possible pairs
        self.pairs = []
        self.labels = []
        self._create_pairs()
        
    def _create_pairs(self):
        """Create all attribute-concept pairs with labels."""
        for _, attr_row in self.attributes_df.iterrows():
            attribute_text = f"{attr_row['attribute_name']} {attr_row['description']}"
            
            # For each concept definition
            for _, concept_row in self.concepts_df.iterrows():
                concept_text = f"{concept_row['domain']}-{concept_row['concept']}: {concept_row['concept_definition']}"
                
                # Label is 1 if domain and concept match
                label = 1.0 if (attr_row['domain'] == concept_row['domain'] and 
                               attr_row['concept'] == concept_row['concept']) else 0.0
                
                self.pairs.append((attribute_text, concept_text))
                self.labels.append(label)
        
        self.pairs = np.array(self.pairs)
        self.labels = np.array(self.labels)
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        return (self.pairs[idx][0], self.pairs[idx][1], self.labels[idx])

class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.similarity = nn.CosineSimilarity(dim=1)
        
    def forward(self, text1: List[str], text2: List[str]) -> torch.Tensor:
        # Encode both texts
        embeddings1 = self.encoder.encode(text1, convert_to_tensor=True)
        embeddings2 = self.encoder.encode(text2, convert_to_tensor=True)
        
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Calculate similarity
        similarity = self.similarity(embeddings1, embeddings2)
        
        return similarity

class SiameseLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Binary cross-entropy with logits
        return F.binary_cross_entropy_with_logits(predictions, targets)

class DataProcessor:
    def __init__(self, attributes_path: str, concepts_path: str):
        """Initialize DataProcessor with paths to CSV files."""
        self.attributes_path = Path(attributes_path)
        self.concepts_path = Path(concepts_path)
        self.logger = logging.getLogger(__name__)
        
        # Load and validate data
        self.attributes_df = None
        self.concepts_df = None
        self.domain_concepts = {}
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
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name} CSV: {missing_cols}")

    def _build_domain_concepts_cache(self) -> None:
        for _, row in self.concepts_df.iterrows():
            if row['domain'] not in self.domain_concepts:
                self.domain_concepts[row['domain']] = set()
            self.domain_concepts[row['domain']].add(row['concept'])
        
        self.logger.info(f"Found {len(self.domain_concepts)} domains with unique concepts")

    def get_all_concept_texts(self) -> List[str]:
        """Get list of all concept texts."""
        return [
            f"{row['domain']}-{row['concept']}: {row['concept_definition']}"
            for _, row in self.concepts_df.iterrows()
        ]

class ModelTrainer:
    def __init__(self, 
                model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                batch_size: int = 32,
                num_epochs: int = 10,
                learning_rate: float = 2e-5,
                margin: float = 0.5,
                output_dir: str = './model_output'):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.margin = margin
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = SiameseNetwork(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self, data_processor: DataProcessor) -> None:
        try:
            self.logger.info("Preparing training data...")
            
            # Initialize dataset and dataloader
            dataset = SiameseDataset(
                data_processor.attributes_df,
                data_processor.concepts_df,
                self.batch_size
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            # Initialize loss and optimizer
            criterion = SiameseLoss(self.margin)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            # Initialize metrics tracker
            metrics_tracker = MetricsTracker(self.output_dir)
            
            self.logger.info(f"Starting training for {self.num_epochs} epochs...")
            
            for epoch in range(self.num_epochs):
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                with tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
                    for batch_text1, batch_text2, batch_labels in pbar:
                        # Move labels to device
                        batch_labels = batch_labels.float().to(self.device)
                        
                        # Forward pass
                        similarities = self.model(batch_text1, batch_text2)
                        
                        # Calculate loss
                        loss = criterion(similarities, batch_labels)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        total_loss += loss.item()
                        predictions = (similarities > 0.5).float()
                        correct_predictions += (predictions == batch_labels).sum().item()
                        total_predictions += len(batch_labels)
                        
                        pbar.set_postfix({
                            'loss': total_loss / (pbar.n + 1),
                            'acc': correct_predictions / total_predictions
                        })
                
                # Calculate epoch metrics
                avg_loss = total_loss / len(dataloader)
                accuracy = correct_predictions / total_predictions
                
                metrics_tracker.add_metric('loss', avg_loss, epoch)
                metrics_tracker.add_metric('accuracy', accuracy, epoch)
                
                # Create training stats
                stats = TrainingStats(
                    epoch=epoch,
                    loss=avg_loss,
                    num_samples=len(dataset),
                    accuracy=accuracy,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size
                )
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}"
                )
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
                    self.save_model(save_path)
                    self.logger.info(f"Saved checkpoint to {save_path}")
            
            # Save final model and metrics
            final_path = os.path.join(self.output_dir, "final-model")
            self.save_model(final_path)
            metrics_tracker.save_metrics()
            self.logger.info(f"Saved final model to {final_path}")
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise
            
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        os.makedirs(path, exist_ok=True)
        self.model.encoder.save(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'siamese_state.pt'))

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
            self.model = SiameseNetwork()
            self.model.encoder = SentenceTransformer(model_path)
            state_dict = torch.load(os.path.join(model_path, 'siamese_state.pt'))
            self.model.load_state_dict(state_dict)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        # Cache concept texts
        self.concept_texts = self.data_processor.get_all_concept_texts()
        
    def predict_single(self, attribute_name: str, description: str) -> List[PredictionResult]:
        """Predict domain-concept for a single attribute."""
        try:
            # Combine attribute text
            attribute_text = f"{attribute_name} {description}"
            
            with torch.no_grad():
                # Get similarities with all concept texts
                similarities = self.model([attribute_text] * len(self.concept_texts), 
                                       self.concept_texts)
# Convert similarities to list
                similarities = similarities.cpu().numpy()
                
                # Create results
                results = []
                for i, concept_text in enumerate(self.concept_texts):
                    # Parse domain and concept from text
                    domain_concept = concept_text.split(':')[0].strip()
                    domain, concept = domain_concept.split('-')
                    
                    results.append(PredictionResult(
                        domain=domain,
                        concept=concept,
                        confidence=float(similarities[i])
                    ))
                
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
    parser = argparse.ArgumentParser(description='Train or predict with Siamese Network model')
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
            trainer = ModelTrainer(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                output_dir=args.output_dir
            )
            
            trainer.train(data_processor)
            
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
