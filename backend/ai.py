"""
AI Model Integration
Calls external AI API using credentials from environment variables.
"""

import os
import json
import requests


def call_ai_model(input_data: dict) -> str:
    """
    Send input_data to OpenAI API and return a conversational response.

    Args:
        input_data: Dictionary containing user input (e.g., from LAST_PREDICT_INPUT)

    Returns:
        Conversational string with AI evacuation guidance, or error message.

    Environment Variables Expected:
        - OPENAI_API_KEY: OpenAI API key (required)
        - OPENAI_MODEL: (Optional) Model identifier; defaults to gpt-4o
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    api_url = "https://api.openai.com/v1/chat/completions"

    if not api_key:
        return "Error: Missing OPENAI_API_KEY environment variable"

    try:
        # Prepare the payload for OpenAI
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a fire evacuation assistant. Respond in exactly this format:

FIRE RISK LEVEL: [CRITICAL/HIGH/MODERATE/LOW]

EVACUATION ROUTE:
[Specific directions in 80 words or less]

SPECIAL ACCOMMODATIONS:
[Any needed accommodations based on their needs, or "None needed" if N/A]

Be direct, clear, and use their provided information (disabilities, pets, kids, medications, concerns).""",
                },
                {
                    "role": "user",
                    "content": f"""I need evacuation guidance for:

Location: {input_data.get('latitude', 'N/A')}°, {input_data.get('longitude', 'N/A')}°
Has disabilities: {input_data.get('has_disability', False)}
Has pets: {input_data.get('has_pets', False)}
Has kids: {input_data.get('has_kids', False)}
Taking medications: {input_data.get('has_medications', False)}
Other concerns: {input_data.get('other_concerns', 'None')}

Provide my evacuation plan in the requested format.""",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        # Add authorization header
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Make the request
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        # Extract the assistant's message text
        message_content = result["choices"][0]["message"]["content"].strip()
        return message_content

    except requests.exceptions.Timeout:
        return "Error: The AI service took too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not reach the AI service. {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error: Could not understand the AI response. {str(e)}"
    except Exception as e:
        return f"Error: Something went wrong. {str(e)}"

