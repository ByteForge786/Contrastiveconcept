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
