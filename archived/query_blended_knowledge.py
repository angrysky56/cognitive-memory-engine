    async def query_blended_knowledge(
        self,
        query: str,
        include_formal: bool = True,
        include_conversational: bool = True
    ) -> dict[str, Any]:
        """
        Unified query interface combining both knowledge tracks.

        Uses semantic similarity for better matching across both
        conversation and document knowledge.

        Returns:
        - formal_knowledge: Structured info from document RTMs
        - conversation_insights: Context from dialogue RTMs
        - cross_references: Links between the tracks
        - unified_summary: Blended understanding

        Args:
            query: Natural language query
            include_formal: Include formal document knowledge
            include_conversational: Include conversation insights

        Returns:
            BlendedQueryResult with combined knowledge
        """
        if not self.initialized:
            await self.initialize()

        self._ensure_component(self.document_store, "document_store")
        self._ensure_component(self.cross_reference_store, "cross_reference_store")

        logger.info(f"Blended query: '{query[:50]}...'")

        result = {
            'query': query,
            'formal_knowledge': [],
            'conversation_insights': {},
            'cross_references': [],
            'unified_summary': '',
            'confidence_score': 0.0
        }

        try:
            # Import semantic similarity calculator
            from ..semantic.similarity_calculator import SemanticSimilarityCalculator
            similarity_calc = SemanticSimilarityCalculator(self.config.embedding_model)
            
            # Track 1: Query formal document knowledge with semantic similarity
            if include_formal:
                all_documents = await self.document_store.get_all_documents()
                formal_matches = []

                # Collect all concepts with their documents
                all_concepts = []
                for doc in all_documents:
                    for concept_id, concept in doc.concepts.items():
                        all_concepts.append((concept, doc))
                
                # Find semantically similar concepts
                for concept, doc in all_concepts:
                    # Calculate semantic similarity
                    concept_text = f"{concept.name} {concept.description} {concept.content}"
                    similarity_score = similarity_calc.calculate_similarity(query, concept_text)
                    
                    if similarity_score > 0.3:  # Threshold for relevance
                        formal_matches.append({
                            'concept_id': concept.concept_id,
                            'document_id': doc.doc_id,
                            'concept': concept,
                            'relevance_score': similarity_score,
                            'document_title': doc.root_concept,
                            'concept_name': concept.name,
                            'description': concept.description[:200] + "..." if len(concept.description) > 200 else concept.description
                        })

                # Sort by relevance and take top matches
                formal_matches.sort(key=lambda x: x['relevance_score'], reverse=True)
                result['formal_knowledge'] = formal_matches[:5]

            # Track 2: Query conversation memory
            if include_conversational:
                conversation_results = await self.query_memory(
                    query=query,
                    context_depth=3,
                    max_results=5
                )

                result['conversation_insights'] = {
                    'results': conversation_results.get('results', []),
                    'context_summary': conversation_results.get('context_summary', ''),
                    'total_results': conversation_results.get('total_results', 0)
                }

            # Track 3: Retrieve persisted cross-references
            if result['formal_knowledge'] or result['conversation_insights']['results']:
                # Get all cross-references with high confidence
                all_links = await self.cross_reference_store.get_all_links(min_confidence=0.5)
                
                # Filter links relevant to our query results
                relevant_links = []
                
                # Check formal knowledge matches
                for match in result['formal_knowledge']:
                    concept_id = match['concept_id']
                    # Find links for this concept
                    concept_links = [link for link in all_links if link.document_concept_id == concept_id]
                    
                    for link in concept_links[:2]:  # Top 2 links per concept
                        relevant_links.append({
                            'link_id': link.link_id,
                            'formal_concept': match['concept_name'],
                            'conversation_fragment': link.context_snippet,
                            'relationship': link.relationship_type.value,
                            'confidence': link.confidence_score,
                            'conversation_id': link.conversation_tree_id
                        })
                
                # Sort by confidence and limit
                relevant_links.sort(key=lambda x: x['confidence'], reverse=True)
                result['cross_references'] = relevant_links[:10]

            # Generate unified summary
            summary_parts = []

            if result['formal_knowledge']:
                formal_summary = f"Formal knowledge: Found {len(result['formal_knowledge'])} relevant concepts"
                if result['formal_knowledge']:
                    top_concept = result['formal_knowledge'][0]
                    formal_summary += f" including '{top_concept['concept_name']}' (relevance: {top_concept['relevance_score']:.2f})"
                summary_parts.append(formal_summary)

            if result['conversation_insights']['results']:
                conv_count = len(result['conversation_insights']['results'])
                conv_summary = f"Conversation insights: Found {conv_count} relevant discussion fragments"
                summary_parts.append(conv_summary)

            if result['cross_references']:
                cross_summary = f"Cross-references: {len(result['cross_references'])} connections between formal knowledge and conversations"
                summary_parts.append(cross_summary)

            result['unified_summary'] = '. '.join(summary_parts) if summary_parts else "No relevant knowledge found"

            # Calculate overall confidence using semantic scores
            formal_confidence = max([m['relevance_score'] for m in result['formal_knowledge']], default=0.0)
            conv_confidence = 0.6 if result['conversation_insights']['results'] else 0.0
            cross_confidence = max([l['confidence'] for l in result['cross_references']], default=0.0)

            result['confidence_score'] = max(formal_confidence, conv_confidence, cross_confidence)

            logger.info(f"Blended query complete: {len(result['formal_knowledge'])} formal, {len(result['conversation_insights'].get('results', []))} conversational, {len(result['cross_references'])} cross-refs")

            return result

        except Exception as e:
            logger.error(f"Error in blended query: {e}")
            result['unified_summary'] = f"Error processing query: {str(e)}"
            return result
