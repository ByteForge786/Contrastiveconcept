def _sample_hard_negatives(self, domain: str, concept: str, 
                         positive_attrs: pd.DataFrame, n_required: int,
                         attributes_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
    hard_negatives = []
    
    # 1. Same domain, different concepts (current approach)
    other_concepts = [c for c in self.data_processor.get_concepts_for_domain(domain) 
                     if c != concept]
    
    # Get potential negatives from same domain
    same_domain_mask = (attributes_df['domain'] == domain) & \
                      (attributes_df['concept'].isin(other_concepts))
    same_domain_negatives = attributes_df[same_domain_mask]
    
    # 2. Different domain, different concepts (new addition)
    other_domains = [d for d in self.data_processor.get_domains() if d != domain]
    diff_domain_mask = (attributes_df['domain'].isin(other_domains))
    diff_domain_negatives = attributes_df[diff_domain_mask]
    
    # Combine all potential negatives
    potential_negatives = pd.concat([same_domain_negatives, diff_domain_negatives])
    
    # Compute similarities in parallel for all negatives
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
    
    # Filter for high similarity (>0.8) regardless of domain
    high_sim_mask = np.array(similarities) > 0.8
    high_sim_negatives = potential_negatives.iloc[high_sim_mask]
    
    # Add hard negatives with domain-based weights
    for _, row in high_sim_negatives.iterrows():
        neg_text = self.data_processor.get_attribute_text(row)
        neg_def = self.data_processor.get_concept_definition(row['domain'], row['concept'])
        
        # Higher weight (1.5) for same domain, slightly lower (1.3) for different domain
        weight = 1.5 if row['domain'] == domain else 1.3
        hard_negatives.append((neg_text, neg_def, weight))
        
        if len(hard_negatives) >= n_required:
            break
    
    return hard_negatives
