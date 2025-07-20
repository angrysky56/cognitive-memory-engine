"""
Response Generator

Implements the Production Module for short-timescale response generation
using local LLMs via Ollama with structured context and social modulation.

This is the "mouth" of the asymmetric architecture - focused, adaptive,
and contextually appropriate response generation.
"""

import time
from datetime import datetime
from typing import Any

from ..llm_providers import LLMProviderFactory
from ..types import GeneratedResponse, RetrievalContext


class ResponseGenerator:
    """
    Generates contextually appropriate responses using local LLMs.

    Implements the Production Module principles:
    - Short-timescale processing (fast, focused generation)
    - Structured prompt formatting with prioritized context
    - Social governance integration (trust, formality, emotion)
    - Predictive error tracking for learning
    """

    def __init__(
        self,
        cloud_config=None,
        max_response_length: int = 10000,  # Increased from artificial 1000 limit
        temperature: float = 0.7
    ):
        self.cloud_config = cloud_config
        self.max_response_length = max_response_length
        self.temperature = temperature

        # Create LLM provider using factory
        self.provider = LLMProviderFactory.create(self.cloud_config)
        self.llm_model = self.provider.get_current_model()

        # Response generation templates
        self.prompt_templates = {
            "default": self._build_default_prompt,
            "technical": self._build_technical_prompt,
            "creative": self._build_creative_prompt,
            "analytical": self._build_analytical_prompt
        }

        # Track generation statistics
        self.generation_stats = {
            "total_responses": 0,
            "avg_generation_time": 0.0,
            "context_utilization": 0.0
        }

    async def _call_llm(self, prompt: str, temperature: float | None = None) -> str:
        """Call the configured LLM provider with the given prompt."""
        temp = temperature if temperature is not None else self.temperature
        return await self.provider.generate(prompt, temperature=temp)

    async def generate_response(
        self,
        query: str,
        context: RetrievalContext,
        include_social_context: bool = True,
        response_type: str = "default"
    ) -> GeneratedResponse:
        """
        Generate response using prioritized context and social modulation.

        Args:
            query: User's question or request
            context: Assembled context from memory systems
            include_social_context: Whether to apply social parameters
            response_type: Type of response (default, technical, creative, analytical)

        Returns:
            GeneratedResponse with content and metadata
        """
        start_time = time.time()

        # Step 1: Build structured prompt
        prompt_builder = self.prompt_templates.get(response_type, self._build_default_prompt)
        structured_prompt = prompt_builder(query, context, include_social_context)

        # Step 2: Generate response using LLM
        raw_response = await self._call_llm(structured_prompt)

        # Step 3: Post-process and validate response
        processed_response = self._post_process_response(raw_response, query, context)

        # Step 4: Track generation metrics
        generation_time = int((time.time() - start_time) * 1000)

        # Step 5: Build response object
        response = GeneratedResponse(
            content=processed_response,
            confidence=self._calculate_confidence(processed_response, context),
            context_nodes_used=len(context.retrieved_nodes),
            context_salience_threshold=min(node.salience_score for node in context.retrieved_nodes) if context.retrieved_nodes else 0.0,
            trust_adjusted=include_social_context and context.trust_score < 1.0,
            formality_adjusted=include_social_context and context.formality_level != 0.5,
            reasoning_chain=context.reasoning_chain.copy(),
            model_used=self.llm_model,
            generation_time_ms=generation_time
        )

        # Update statistics
        self._update_generation_stats(generation_time, context)

        return response

    def _build_default_prompt(
        self,
        query: str,
        context: RetrievalContext,
        include_social_context: bool
    ) -> str:
        """Build a default structured prompt"""
        prompt_parts = []

        # System instruction with cognitive context
        prompt_parts.append("""You are an AI assistant with access to hierarchical memory from previous conversations. Your memory is organized using Random Tree Models (RTM) that compress information at different levels of detail, similar to human memory.

Key principles:
- Use the prioritized context below, where higher salience scores indicate more relevant information
- Provide responses that are contextually grounded in the memory
- Acknowledge when information is uncertain or missing
- Maintain conversational continuity across sessions""")

        # Social context modulation
        if include_social_context:
            social_instructions = self._build_social_instructions(context)
            if social_instructions:
                prompt_parts.append(social_instructions)

        # Prioritized context section
        if context.retrieved_nodes:
            prompt_parts.append("## Relevant Memory Context (Ordered by Salience)")

            for i, node in enumerate(context.retrieved_nodes[:10], 1):  # Top 10 nodes
                salience_indicator = "🔥" if node.salience_score > 2.0 else "⭐" if node.salience_score > 1.5 else "•"

                prompt_parts.append(f"""
{salience_indicator} Context {i} (Salience: {node.salience_score:.2f})
Summary: {node.summary}
Content: {node.content}
Temporal Context: {node.temporal_scale.value} level, {self._format_relative_time(node.timestamp)}
""")

        # Temporal scope information
        if context.temporal_scope:
            prompt_parts.append(f"## Temporal Focus: {context.temporal_scope.value.title()} level information")

        if context.conversation_history_depth > 0:
            prompt_parts.append(f"## Session Context: {context.conversation_history_depth} related conversations in memory")

        # User query
        prompt_parts.append(f"## User Query\n{query}")

        # Response instructions
        prompt_parts.append("""
## Response Instructions
Based on the prioritized memory context above:
1. Provide a helpful, contextually grounded response
2. Reference specific details from memory when relevant
3. If the memory doesn't contain enough information, acknowledge this honestly
4. Maintain appropriate tone based on social context
5. Provide comprehensive responses - be thorough and complete in your explanations

Response:""")

        return "\n".join(prompt_parts)

    def _build_technical_prompt(
        self,
        query: str,
        context: RetrievalContext,
        include_social_context: bool
    ) -> str:
        """Build a technical/analytical prompt"""
        prompt_parts = []

        prompt_parts.append("""You are a technical AI assistant with access to detailed project memory. Focus on providing precise, actionable technical information based on the memory context.""")

        # Technical context emphasis
        if context.retrieved_nodes:
            prompt_parts.append("## Technical Memory Context")

            technical_nodes = [
                node for node in context.retrieved_nodes
                if any(tech_word in (node.content + node.summary).lower()
                      for tech_word in ["api", "code", "bug", "system", "project", "timeline", "technical"])
            ]

            for i, node in enumerate(technical_nodes[:8], 1):
                prompt_parts.append(f"""
📋 Technical Context {i} (Salience: {node.salience_score:.2f})
{node.summary}
Details: {node.content}
""")

        prompt_parts.append(f"## Technical Query\n{query}")

        prompt_parts.append("""
## Technical Response Guidelines
- Provide specific, actionable technical information
- Reference exact details from memory when available
- Include relevant timelines, dependencies, or constraints mentioned in memory
- Suggest concrete next steps when appropriate
- Be precise about what information is available vs. missing

Technical Response:""")

        return "\n".join(prompt_parts)

    def _build_creative_prompt(
        self,
        query: str,
        context: RetrievalContext,
        include_social_context: bool
    ) -> str:
        """Build a creative/ideation prompt"""
        prompt_parts = []

        prompt_parts.append("""You are a creative AI assistant that draws inspiration from memory to generate novel ideas. Use the memory context as a foundation for creative thinking and ideation.""")

        if context.retrieved_nodes:
            prompt_parts.append("## Inspirational Memory Context")

            for i, node in enumerate(context.retrieved_nodes[:6], 1):
                prompt_parts.append(f"""
💡 Inspiration {i}: {node.summary}
Context: {node.content}
""")

        prompt_parts.append(f"## Creative Challenge\n{query}")

        prompt_parts.append("""
## Creative Response Guidelines
- Build upon themes and concepts from memory
- Generate novel connections between remembered information
- Encourage exploration and experimentation
- Provide multiple perspectives or approaches when possible
- Balance creativity with practical grounding

Creative Response:""")

        return "\n".join(prompt_parts)

    def _build_analytical_prompt(
        self,
        query: str,
        context: RetrievalContext,
        include_social_context: bool
    ) -> str:
        """Build an analytical/research prompt"""
        prompt_parts = []

        prompt_parts.append("""You are an analytical AI assistant that synthesizes information from memory to provide insights and analysis. Focus on patterns, relationships, and deeper understanding.""")

        if context.retrieved_nodes:
            prompt_parts.append("## Memory Evidence Base")

            # Group nodes by themes for analysis
            themed_content = {}
            for node in context.retrieved_nodes:
                # Simple theme extraction
                if "project" in node.content.lower():
                    themed_content.setdefault("Projects", []).append(node)
                elif any(word in node.content.lower() for word in ["problem", "issue", "blocker"]):
                    themed_content.setdefault("Challenges", []).append(node)
                elif any(word in node.content.lower() for word in ["solution", "recommend", "suggest"]):
                    themed_content.setdefault("Solutions", []).append(node)
                else:
                    themed_content.setdefault("General", []).append(node)

            for theme, nodes in themed_content.items():
                prompt_parts.append(f"\n### {theme} ({len(nodes)} memories)")
                for node in nodes[:3]:  # Top 3 per theme
                    prompt_parts.append(f"- {node.summary} (Salience: {node.salience_score:.2f})")

        prompt_parts.append(f"## Analysis Request\n{query}")

        prompt_parts.append("""
## Analytical Response Guidelines
- Identify patterns and relationships in the memory
- Synthesize information across different memories
- Provide evidence-based insights
- Acknowledge limitations in the available information
- Structure analysis clearly with supporting details

Analysis:""")

        return "\n".join(prompt_parts)

    def _build_social_instructions(self, context: RetrievalContext) -> str:
        """Build social context instructions"""
        instructions = []

        # Trust level modulation
        if context.trust_score < 0.7:
            instructions.append("- Use a more cautious, professional tone")
            instructions.append("- Provide more detailed explanations and citations")
        elif context.trust_score > 0.9:
            instructions.append("- Use a more conversational, familiar tone")
            instructions.append("- Feel free to be more direct and concise")

        # Formality level modulation
        if context.formality_level > 0.8:
            instructions.append("- Maintain formal, professional language")
            instructions.append("- Use complete sentences and proper structure")
        elif context.formality_level < 0.3:
            instructions.append("- Use casual, conversational language")
            instructions.append("- Feel free to use contractions and informal expressions")

        # Emotional context
        if context.emotional_context:
            if context.emotional_context.get("engagement", 0) > 0.8:
                instructions.append("- Show enthusiasm and engagement with the topic")
            if context.emotional_context.get("curiosity", 0) > 0.8:
                instructions.append("- Encourage exploration and ask follow-up questions")

        if instructions:
            return "## Social Context Guidelines\n" + "\n".join(instructions) + "\n"

        return ""



    def _post_process_response(
        self,
        raw_response: str,
        query: str,
        context: RetrievalContext
    ) -> str:
        """Post-process the generated response"""
        if not raw_response:
            return "I apologize, but I was unable to generate a response to your query."

        # Remove any unwanted artifacts
        processed = raw_response.strip()

        # Note: Removed artificial length truncation - let the LLM provide full responses

        # Add context utilization note if helpful
        if context.retrieved_nodes and context.max_salience_score > 2.0:
            processed += f"\n\n*Response based on {len(context.retrieved_nodes)} relevant memories with high contextual relevance.*"

        return processed

    def _calculate_confidence(
        self,
        response: str,
        context: RetrievalContext
    ) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5  # Base confidence

        # Boost confidence based on context quality
        if context.retrieved_nodes:
            avg_salience = context.avg_salience_score
            confidence += min(0.3, avg_salience / 10.0)  # Max +0.3 from salience

        # Boost confidence based on response completeness and structure
        if len(response) > 100 and '.' in response:
            confidence += 0.1

        # Note: Removed penalty for short responses - sometimes concise answers are appropriate

        # Boost confidence for responses that reference context
        context_words = set()
        for node in context.retrieved_nodes[:5]:
            context_words.update(node.summary.lower().split())

        response_words = set(response.lower().split())
        context_overlap = len(context_words.intersection(response_words))

        if context_overlap > 3:
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def _format_relative_time(self, timestamp: datetime) -> str:
        """Format timestamp as relative time"""
        delta = datetime.now() - timestamp

        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hours ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "just now"

    def _update_generation_stats(self, generation_time: int, context: RetrievalContext):
        """Update generation statistics"""
        self.generation_stats["total_responses"] += 1

        # Running average of generation time
        total = self.generation_stats["total_responses"]
        old_avg = self.generation_stats["avg_generation_time"]
        self.generation_stats["avg_generation_time"] = (
            (old_avg * (total - 1) + generation_time) / total
        )

        # Context utilization metric
        if context.retrieved_nodes:
            utilization = len(context.retrieved_nodes) / 20.0  # Normalize to 20 max nodes
            old_util = self.generation_stats["context_utilization"]
            self.generation_stats["context_utilization"] = (
                (old_util * (total - 1) + utilization) / total
            )

    async def get_generation_statistics(self) -> dict[str, Any]:
        """Get response generation statistics"""
        return self.generation_stats.copy()

    async def update_temperature(self, new_temperature: float):
        """Update generation temperature for different response styles"""
        self.temperature = max(0.1, min(1.0, new_temperature))

    async def predict_response_quality(
        self,
        query: str,
        context: RetrievalContext
    ) -> dict[str, float]:
        """
        Predict response quality before generation.

        This implements part of the predictive modulator concept.
        """
        predictions = {
            "relevance": 0.5,
            "completeness": 0.5,
            "accuracy": 0.5
        }

        # Predict relevance based on context salience
        if context.retrieved_nodes:
            predictions["relevance"] = min(1.0, context.max_salience_score / 3.0)

        # Predict completeness based on context breadth
        if len(context.retrieved_nodes) >= 5:
            predictions["completeness"] = 0.8
        elif len(context.retrieved_nodes) >= 2:
            predictions["completeness"] = 0.6

        # Predict accuracy based on temporal recency
        if context.retrieved_nodes:
            recent_nodes = [
                node for node in context.retrieved_nodes
                if (datetime.now() - node.timestamp).days < 7
            ]
            predictions["accuracy"] = len(recent_nodes) / len(context.retrieved_nodes)

        return predictions
