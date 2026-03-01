"""
AI Model Integration
Calls external AI API using credentials from environment variables.
"""

import os
import json
import requests


def call_ai_model(input_data: dict) -> dict:
    """
    Send input_data to OpenAI API and return the response.

    Args:
        input_data: Dictionary containing user input (e.g., from LAST_PREDICT_INPUT)

    Returns:
        Dictionary with AI model response or error details.

    Environment Variables Expected:
        - OPENAI_API_KEY: OpenAI API key (required)
        - OPENAI_MODEL: (Optional) Model identifier; defaults to gpt-4o
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    api_url = "https://api.openai.com/v1/chat/completions"

    if not api_key:
        return {
            "error": "Missing OPENAI_API_KEY environment variable",
            "status": "error",
        }

    try:
        # Prepare the payload for OpenAI
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fire evacuation assistant providing guidance based on user input and location.",
                },
                {
                    "role": "user",
                    "content": f"""Based on the following user input, provide structured evacuation guidance.

User Input:
{json.dumps(input_data, indent=2)}

Provide your response in this exact format:
FIRE_RISK: [LOW|MODERATE|HIGH|CRITICAL]
EVACUATION_ROUTE: [Provide specific directions]
SPECIAL_ACCOMMODATIONS: [List any needed accommodations]
""",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500,
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
        return {
            "status": "success",
            "data": result,
            "raw_input": input_data,
        }

    except requests.exceptions.Timeout:
        return {
            "error": "AI model request timed out",
            "status": "error",
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"AI model request failed: {str(e)}",
            "status": "error",
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse AI response: {str(e)}",
            "status": "error",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error calling AI model: {str(e)}",
            "status": "error",
        }


def get_ai_response_text(ai_result: dict) -> str:
    """
    Extract readable text from AI model response.

    Args:
        ai_result: Dictionary returned by call_ai_model()

    Returns:
        Formatted string with AI response or error message
    """
    if ai_result.get("status") == "error":
        return f"Error: {ai_result.get('error', 'Unknown error')}"

    try:
        data = ai_result.get("data", {})
        # Extract from OpenAI-style response
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            return message.get("content", "No response content")
        # Fallback for other response formats
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Failed to extract response: {str(e)}"
