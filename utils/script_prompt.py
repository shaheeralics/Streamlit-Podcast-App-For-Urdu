"""
Script generation prompts for OpenAI to create conversational podcast scripts.
Generates structured prompts for Australian-style conversational content.
"""

from typing import List, Dict, Any

def build_messages(
    article_title: str,
    article_text: str,
    host_name: str = "Alex",
    guest_name: str = "Sarah",
    aussie: bool = True
) -> List[Dict[str, str]]:
    """
    Build OpenAI chat messages for podcast script generation
    
    Args:
        article_title: Title of the article
        article_text: Main content of the article
        host_name: Name of the podcast host
        guest_name: Name of the podcast guest
        aussie: Whether to use Australian conversational style
        
    Returns:
        List of OpenAI chat message dictionaries
    """
    
    style_instruction = _get_style_instruction(aussie)
    system_prompt = _build_system_prompt(host_name, guest_name, style_instruction)
    user_prompt = _build_user_prompt(article_title, article_text, host_name, guest_name)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def _get_style_instruction(aussie: bool) -> str:
    """Get style-specific instructions"""
    if aussie:
        return """
Use Australian conversational style with these characteristics:
- Natural, relaxed tone with Australian expressions and idioms
- Use words like "mate", "fair dinkum", "crikey", "bloody", "heaps", "reckon"
- Casual contractions: "can't", "won't", "she'll be right"
- Australian pronunciation hints in spelling where natural: "no worries", "cheers"
- Friendly, down-to-earth approach to complex topics
- Occasional use of "G'day", "too right", "stone the flamin' crows" for emphasis
- Make it sound like a genuine conversation between Aussie mates discussing interesting news
"""
    else:
        return """
Use natural, conversational style with these characteristics:
- Professional but friendly tone
- Clear, accessible language for general audiences
- Natural speech patterns with appropriate contractions
- Engaging, informative discussion style
- Balance of casual conversation and informative content
"""

def _build_system_prompt(host_name: str, guest_name: str, style_instruction: str) -> str:
    """Build the system prompt for OpenAI"""
    return f"""You are an expert podcast script writer creating engaging conversational content.

Your task is to convert article content into a natural, flowing conversation between two podcast hosts.

CHARACTERS:
- {host_name}: The primary host who introduces topics and guides the conversation
- {guest_name}: The co-host who provides insights, asks questions, and adds commentary

STYLE REQUIREMENTS:
{style_instruction}

CONVERSATION STRUCTURE:
- Start with a warm, engaging introduction to the topic
- Break down complex information into digestible, conversational chunks
- Include natural reactions, questions, and clarifications between hosts
- Add personal anecdotes or relatable examples where appropriate
- Include smooth transitions between different aspects of the topic
- End with a thoughtful conclusion and call-to-action

TECHNICAL REQUIREMENTS:
- Return ONLY valid JSON with no additional text
- JSON structure: {{"script": [list of turns]}}
- Each turn: {{"speaker": "host" or "guest", "text": "spoken content"}}
- Keep individual speaking turns between 20-100 words for natural flow
- Aim for 15-25 total turns for a 5-10 minute podcast
- Ensure the conversation feels authentic and unscripted

CONTENT GUIDELINES:
- Stay factually accurate to the source material
- Make complex topics accessible and engaging
- Include natural speech patterns (pauses, interjections, clarifications)
- Balance information delivery with conversational flow
- Add context and explanations for technical terms"""

def _build_user_prompt(article_title: str, article_text: str, host_name: str, guest_name: str) -> str:
    """Build the user prompt with article content"""
    
    # Limit article text length to avoid token limits
    max_length = 4000
    if len(article_text) > max_length:
        article_text = article_text[:max_length] + "..."
    
    return f"""Create a conversational podcast script based on this article:

ARTICLE TITLE: {article_title}

ARTICLE CONTENT:
{article_text}

Create a natural conversation between {host_name} (host) and {guest_name} (guest) discussing this article. 

The conversation should:
1. Open with {host_name} introducing the topic in an engaging way
2. Have both hosts explore the key points naturally
3. Include questions, reactions, and insights from both perspectives
4. Make the content accessible and interesting for general audiences
5. Close with a thoughtful summary and invitation for audience engagement

Remember to return ONLY the JSON response with no additional text or formatting."""

def validate_script_response(response_text: str, host_name: str = "Alex", guest_name: str = "Sarah") -> Dict[str, Any]:
    """
    Validate and parse OpenAI script response
    
    Args:
        response_text: Raw response from OpenAI
        host_name: Expected host name
        guest_name: Expected guest name
        
    Returns:
        Parsed and validated script dictionary
        
    Raises:
        Exception: If response is invalid
    """
    import json
    import re
    
    try:
        # Clean the response text - remove markdown code blocks
        cleaned_text = response_text.strip()
        
        # Remove ```json and ``` markers if present
        if cleaned_text.startswith('```'):
            # Find the first newline after ```json or ```
            first_newline = cleaned_text.find('\n')
            if first_newline != -1:
                cleaned_text = cleaned_text[first_newline + 1:]
        
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        
        # Remove any remaining leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON response
        parsed = json.loads(cleaned_text)
        
        # Validate structure
        if not isinstance(parsed, dict):
            raise Exception("Response must be a JSON object")
        
        if "script" not in parsed:
            raise Exception("Response must contain 'script' key")
        
        script = parsed["script"]
        if not isinstance(script, list):
            raise Exception("Script must be a list of turns")
        
        if len(script) == 0:
            raise Exception("Script cannot be empty")
        
        # Normalize speaker names and validate each turn
        normalized_script = []
        for i, turn in enumerate(script):
            if not isinstance(turn, dict):
                raise Exception(f"Turn {i+1} must be an object")
            
            if "speaker" not in turn or "text" not in turn:
                raise Exception(f"Turn {i+1} must have 'speaker' and 'text' fields")
            
            speaker = turn["speaker"].strip()
            text = turn["text"].strip()
            
            # Normalize speaker names to host/guest
            if speaker.lower() == host_name.lower():
                normalized_speaker = "host"
            elif speaker.lower() == guest_name.lower():
                normalized_speaker = "guest"
            elif speaker.lower() in ["host"]:
                normalized_speaker = "host"
            elif speaker.lower() in ["guest"]:
                normalized_speaker = "guest"
            else:
                # Try to guess based on common patterns
                if any(name in speaker.lower() for name in [host_name.lower(), "host", "alex"]):
                    normalized_speaker = "host"
                elif any(name in speaker.lower() for name in [guest_name.lower(), "guest", "sarah"]):
                    normalized_speaker = "guest"
                else:
                    raise Exception(f"Turn {i+1}: unknown speaker '{speaker}'. Expected '{host_name}' (host) or '{guest_name}' (guest)")
            
            if not text:
                raise Exception(f"Turn {i+1}: text cannot be empty")
            
            if len(text) < 10:
                raise Exception(f"Turn {i+1}: text too short (minimum 10 characters)")
            
            normalized_script.append({
                "speaker": normalized_speaker,
                "text": text
            })
        
        # Return the normalized script
        return {"script": normalized_script}
        
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        if "Response must" in str(e) or "Turn" in str(e) or "Script" in str(e):
            raise e
        raise Exception(f"Error validating script: {str(e)}")

def get_sample_script() -> List[Dict[str, str]]:
    """
    Get a sample script for testing purposes
    
    Returns:
        Sample script in the correct format
    """
    return [
        {
            "speaker": "host",
            "text": "G'day everyone! Welcome back to the podcast. I'm Alex, and today we've got some fascinating news to dive into with my co-host Sarah."
        },
        {
            "speaker": "guest", 
            "text": "Thanks Alex! I'm really excited about today's topic. This article caught my attention because it touches on something we've all been thinking about lately."
        },
        {
            "speaker": "host",
            "text": "Too right! So let's jump straight in. The main story here is about how technology is changing the way we work and live. Sarah, what was your first reaction when you read this?"
        },
        {
            "speaker": "guest",
            "text": "Well mate, I reckon it's both exciting and a bit concerning. On one hand, we're seeing incredible innovations that could make life easier. But there's also this question of how it affects jobs and privacy."
        },
        {
            "speaker": "host",
            "text": "That's a fair point. The article mentions some specific examples that really drive home your point. Let's break down what the research actually found..."
        }
    ]
