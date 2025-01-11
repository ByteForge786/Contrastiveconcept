def prepare_batches(self,
                  data_processor: OptimizedDataProcessor,
                  sampler: OptimizedPairSampler,
                  is_training: bool = True,
                  batch_start_idx: int = 0) -> List[TrainingBatch]:
    """
    Prepare batches for training or validation with optimized parallel processing.
    
    Args:
        data_processor: OptimizedDataProcessor instance
        sampler: OptimizedPairSampler instance
        is_training: Boolean indicating training or validation mode
        batch_start_idx: Starting index for batch processing
        
    Returns:
        List[TrainingBatch]: List of prepared training batches
    """
    try:
        batches = []
        # Get all domain-concept pairs
        domain_concepts = []
        for domain in data_processor.get_domains():
            concepts = data_processor.get_concepts_for_domain(domain)
            domain_concepts.extend([(domain, concept) for concept in concepts])

        # Calculate batch range for this worker
        batch_end_idx = min(batch_start_idx + self.batch_size, len(domain_concepts))
        worker_pairs = domain_concepts[batch_start_idx:batch_end_idx]
        
        # Create thread pool for parallel sampling
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Process domain-concept pairs in parallel
            future_to_pair = {
                executor.submit(self._process_single_batch, 
                              data_processor,
                              sampler,
                              domain,
                              concept,
                              is_training): (domain, concept)
                for domain, concept in worker_pairs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pair):
                domain, concept = future_to_pair[future]
                try:
                    batch = future.result()
                    if batch is not None:
                        batches.append(batch)
                except Exception as e:
                    self.logger.error(
                        f"Error processing batch for {domain}-{concept}: {str(e)}"
                    )
                    continue

        # Sort batches by size for more efficient processing
        batches.sort(key=lambda x: len(x.anchors), reverse=True)
        
        # Apply batch-level optimizations
        optimized_batches = self._optimize_batches(batches)
        
        return optimized_batches

    except Exception as e:
        self.logger.error(f"Error preparing batches: {str(e)}")
        return []

def _process_single_batch(self,
                        data_processor: OptimizedDataProcessor,
                        sampler: OptimizedPairSampler,
                        domain: str,
                        concept: str,
                        is_training: bool) -> Optional[TrainingBatch]:
    """Process a single domain-concept pair into a batch."""
    try:
        # Sample pairs with caching
        cache_key = (domain, concept, is_training)
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]
            
        result = sampler.sample_pairs_batch(
            domain,
            concept,
            data_processor.attributes_df,
            self.batch_size
        )
        
        if not result.positive_pairs:
            return None
            
        # Create optimized batch
        batch = TrainingBatch(
            anchors=[result.positive_pairs[0][1]] * len(result.positive_pairs),
            positives=[pair[0] for pair in result.positive_pairs],
            negatives=[pair[0] for pair in result.negative_pairs],
            negative_weights=torch.tensor([pair[2] for pair in result.negative_pairs])
        )
        
        # Cache the result
        self._batch_cache[cache_key] = batch
        
        return batch
        
    except Exception as e:
        self.logger.error(
            f"Error processing single batch for {domain}-{concept}: {str(e)}"
        )
        return None

def _optimize_batches(self, batches: List[TrainingBatch]) -> List[TrainingBatch]:
    """Apply batch-level optimizations."""
    if not batches:
        return []
        
    # Combine small batches
    min_batch_size = self.batch_size // 2
    optimized = []
    current_batch = None
    
    for batch in batches:
        if len(batch.anchors) < min_batch_size:
            if current_batch is None:
                current_batch = batch
            else:
                # Combine batches
                current_batch = TrainingBatch(
                    anchors=current_batch.anchors + batch.anchors,
                    positives=current_batch.positives + batch.positives,
                    negatives=current_batch.negatives + batch.negatives,
                    negative_weights=torch.cat([
                        current_batch.negative_weights,
                        batch.negative_weights
                    ])
                )
                
                if len(current_batch.anchors) >= self.batch_size:
                    optimized.append(current_batch)
                    current_batch = None
        else:
            if current_batch is not None:
                optimized.append(current_batch)
                current_batch = None
            optimized.append(batch)
    
    if current_batch is not None:
        optimized.append(current_batch)
        
    return optimized







class PairSampler:
    def __init__(self, data_processor: DataProcessor, batch_size: int = 32):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        # Initialize model once during creation of PairSampler
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def compute_description_similarity(self, desc1: str, desc2: str) -> float:
        # Use the pre-initialized model
        embedding1 = self.model.encode(desc1)
        embedding2 = self.model.encode(desc2)
        similarity = util.cos_sim(embedding1, embedding2)
        return float(similarity[0][0])




class PairSampler:
    def __init__(self, data_processor: DataProcessor, batch_size: int = 32):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def compute_description_similarity(self, desc1: str, desc2: str) -> float:
        embeddings = util.normalize_embeddings(
            util.semantic_search(desc1, [desc2], top_k=1)[0]
        )
        return float(embeddings[0]['score'])

    def _sample_hard_negatives(self, domain: str, concept: str, 
                             positive_attrs: pd.DataFrame, n_required: int,
                             attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
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

    def _sample_medium_negatives(self, domain: str, concept: str, 
                               n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample medium negatives using deterministic selection."""
        medium_negatives = []
        
        # Get all other domains
        other_domains = [d for d in self.data_processor.get_domains() if d != domain]
        
        if not other_domains:
            self.logger.warning("No other domains found for medium negatives")
            return medium_negatives
        
        # Create a deterministic selection of domain-concept pairs
        negative_candidates = []
        for d in other_domains:
            concepts = self.data_processor.get_concepts_for_domain(d)
            for c in concepts:
                # Get negative attributes for this domain-concept pair
                mask = (attributes_df['domain'] == d) & (attributes_df['concept'] == c)
                neg_attrs = attributes_df[mask]
                
                if len(neg_attrs) > 0:
                    # Take the first attribute from each domain-concept pair
                    neg_row = neg_attrs.iloc[0]
                    negative_candidates.append((neg_row, d, c))

        # If we have candidates, take them sequentially until we have enough
        count = 0
        while len(medium_negatives) < n_required and count < len(negative_candidates):
            neg_row, d, c = negative_candidates[count % len(negative_candidates)]
            try:
                neg_text = self.data_processor.get_attribute_text(neg_row)
                neg_def = self.data_processor.get_concept_definition(d, c)
                medium_negatives.append((neg_text, neg_def, 1.0))
            except Exception as e:
                self.logger.warning(f"Error processing negative sample: {str(e)}")
            count += 1

        if len(medium_negatives) < n_required:
            self.logger.warning(
                f"Could only generate {len(medium_negatives)} medium negatives "
                f"out of {n_required} requested"
            )
            
            # If we still need more negatives, duplicate existing ones
            while len(medium_negatives) < n_required:
                idx = len(medium_negatives) % len(negative_candidates)
                medium_negatives.append(medium_negatives[idx])

        return medium_negatives

    def sample_pairs(self, domain: str, concept: str, is_training: bool = True) -> SamplingResult:
        try:
            # Get attributes based on split
            attributes_df = (
                self.data_processor.get_train_attributes() if is_training 
                else self.data_processor.get_val_attributes()
            )
            
            # Get all positive attributes
            mask = (attributes_df['domain'] == domain) & \
                  (attributes_df['concept'] == concept)
            positive_attrs = attributes_df[mask]
            
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
            hard_negatives = self._sample_hard_negatives(
                domain, concept, positive_attrs, n_hard, attributes_df
            )
            
            # Sample medium negatives (60% or remainder)
            n_medium = n_required_negatives - len(hard_negatives)
            medium_negatives = self._sample_medium_negatives(
                domain, concept, n_medium, attributes_df
            )
            
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













import random
from datetime import datetime

class PairSampler:
    def __init__(self, data_processor: DataProcessor, batch_size: int = 32):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        # Set random seed based on current time
        random.seed(datetime.now().timestamp())

    # Your compute_description_similarity and _sample_hard_negatives methods remain same

    def _sample_medium_negatives(self, domain: str, concept: str, 
                               n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Sample medium negatives using simple random sampling."""
        medium_negatives = []
        other_domains = [d for d in self.data_processor.get_domains() if d != domain]
        
        if not other_domains:
            self.logger.warning("No other domains found for medium negatives")
            return medium_negatives

        # Create a pool of all possible negative samples
        negative_pool = []
        for d in other_domains:
            concepts = self.data_processor.get_concepts_for_domain(d)
            for c in concepts:
                mask = (attributes_df['domain'] == d) & (attributes_df['concept'] == c)
                neg_attrs = attributes_df[mask]
                
                if len(neg_attrs) > 0:
                    for _, row in neg_attrs.iterrows():
                        negative_pool.append((row, d, c))

        # If we have candidates in the pool, randomly sample them
        while len(medium_negatives) < n_required and negative_pool:
            # Randomly select an index
            idx = random.randint(0, len(negative_pool) - 1)
            neg_row, d, c = negative_pool[idx]
            
            try:
                neg_text = self.data_processor.get_attribute_text(neg_row)
                neg_def = self.data_processor.get_concept_definition(d, c)
                medium_negatives.append((neg_text, neg_def, 1.0))
                
                # Optionally remove used sample to avoid duplicates
                # negative_pool.pop(idx)  # Uncomment if you want to avoid reusing samples
                
            except Exception as e:
                self.logger.warning(f"Error processing negative sample: {str(e)}")
                negative_pool.pop(idx)  # Remove problematic sample

        # If we still need more negatives, cycle through existing ones
        if len(medium_negatives) < n_required:
            self.logger.warning(
                f"Could only generate {len(medium_negatives)} medium negatives "
                f"out of {n_required} requested. Cycling through existing ones."
            )
            
            while len(medium_negatives) < n_required:
                existing_idx = random.randint(0, len(medium_negatives) - 1)
                medium_negatives.append(medium_negatives[existing_idx])

        return medium_negatives

    def simple_random_sample(self, items: list, n: int, with_replacement: bool = True) -> list:
        """Simple custom random sampling function."""
        if not items:
            return []
            
        if n <= 0:
            return []
            
        if not with_replacement and n > len(items):
            return items
            
        result = []
        available_indices = list(range(len(items)))
        
        while len(result) < n:
            if not available_indices:
                if with_replacement:
                    available_indices = list(range(len(items)))
                else:
                    break
                    
            idx = random.randint(0, len(available_indices) - 1)
            selected_idx = available_indices[idx]
            
            if not with_replacement:
                available_indices.pop(idx)
                
            result.append(items[selected_idx])
            
        return result

    def sample_pairs(self, domain: str, concept: str, is_training: bool = True) -> SamplingResult:
        """Sample positive and negative pairs for training."""
        try:
            # Get attributes based on split
            attributes_df = (
                self.data_processor.get_train_attributes() if is_training 
                else self.data_processor.get_val_attributes()
            )
            
            # Get all positive attributes
            mask = (attributes_df['domain'] == domain) & \
                  (attributes_df['concept'] == concept)
            positive_attrs = attributes_df[mask]
            
            if len(positive_attrs) == 0:
                raise ValueError(f"No attributes found for domain={domain}, concept={concept}")

            concept_def = self.data_processor.get_concept_definition(domain, concept)
            
            # Create positive pairs
            positive_pairs = []
            for _, row in positive_attrs.iterrows():
                try:
                    attr_text = self.data_processor.get_attribute_text(row)
                    positive_pairs.append((attr_text, concept_def))
                except Exception as e:
                    self.logger.warning(f"Error processing positive pair: {str(e)}")
            
            if not positive_pairs:
                raise ValueError(f"No valid positive pairs could be created for {domain}-{concept}")
            
            n_required_negatives = len(positive_pairs)
            
            # Sample hard negatives (40%)
            n_hard = int(0.4 * n_required_negatives)
            hard_negatives = self._sample_hard_negatives(
                domain, concept, positive_attrs, n_hard, attributes_df
            )
            
            # Sample medium negatives (60% or remainder)
            n_medium = n_required_negatives - len(hard_negatives)
            medium_negatives = self._sample_medium_negatives(
                domain, concept, n_medium, attributes_df
            )
            
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






def _sample_medium_negatives(self, domain: str, concept: str, 
                           n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """Sample medium negatives from different domains."""
    medium_negatives = []
    other_domains = [d for d in self.data_processor.get_domains() if d != domain]
    
    if not other_domains:
        self.logger.warning("No other domains found for medium negatives")
        return medium_negatives
        
    # Convert to list to avoid dtype issues
    other_domains = list(other_domains)
    
    # Use rng instead of direct numpy random
    rng = np.random.default_rng()
    sample_domains = rng.choice(other_domains, size=n_required, replace=True)
    
    for d in sample_domains:
        concepts = self.data_processor.get_concepts_for_domain(d)
        if not concepts:
            continue
            
        # Convert concepts to list
        concepts = list(concepts)
        c = rng.choice(concepts)
        
        # Create mask using boolean indexing
        mask = attributes_df['domain'].eq(d) & attributes_df['concept'].eq(c)
        neg_attrs = attributes_df[mask]
        
        if len(neg_attrs) == 0:
            continue
            
        # Use random integer for indexing
        random_idx = rng.integers(len(neg_attrs))
        neg_row = neg_attrs.iloc[random_idx]
        neg_text = self.data_processor.get_attribute_text(neg_row)
        neg_def = self.data_processor.get_concept_definition(d, c)
        
        medium_negatives.append((neg_text, neg_def, 1.0))
        
        if len(medium_negatives) >= n_required:
            break
            
    return medium_negatives





def _sample_medium_negatives(self, domain: str, concept: str, 
                           n_required: int, attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """Sample medium negatives from different domains."""
    medium_negatives = []
    other_domains = [d for d in self.data_processor.get_domains() if d != domain]
    
    if not other_domains:
        self.logger.warning("No other domains found for medium negatives")
        return medium_negatives
        
    # Convert to list and use random.choices instead of np.random.choice
    sample_domains = random.choices(other_domains, k=n_required)
    
    for d in sample_domains:
        concepts = self.data_processor.get_concepts_for_domain(d)
        if not concepts:
            continue
            
        # Convert concepts to list before random choice
        c = random.choice(list(concepts))
        
        mask = (attributes_df['domain'] == d) & \
               (attributes_df['concept'] == c)
        neg_attrs = attributes_df[mask]
        
        if len(neg_attrs) == 0:
            continue
            
        # Use random.randint instead of np.random.randint
        neg_row = neg_attrs.iloc[random.randint(0, len(neg_attrs)-1)]
        neg_text = self.data_processor.get_attribute_text(neg_row)
        neg_def = self.data_processor.get_concept_definition(d, c)
        
        medium_negatives.append((neg_text, neg_def, 1.0))
        
        if len(medium_negatives) >= n_required:
            break
            
    return medium_negatives




def _sample_hard_negatives(self, domain: str, concept: str, 
                         positive_attrs: pd.DataFrame, n_required: int,
                         attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
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
