import re, json
import google.generativeai as genai

def corrected_text(ocr_text):
    # API key
    genai.configure(api_key="AIzaSyCvU6dmzdII3PAgK9Ea2EYtJ8mNgL48Txs")  

    # Gemini model
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Prompt
    prompt = f"""
    You are a dyslexia-aware writing coach.

    The following text was extracted from a student's handwriting using OCR:
    \"\"\"{ocr_text}\"\"\"

    Your tasks:
    1. Detect symptoms of dyslexia (e.g., spelling mistakes, punctuation issues, letter reversals, transpositions, spacing errors, etc.).
    2. Suggest 2–5 helpful strategies to improve the student's writing skills.
    3. Correct the paragraph completely, using proper grammar and **standard capitalization rules**.
    - **Do not write any words in full capital letters unless they are acronyms (e.g., NASA) or proper nouns.**

    Respond ONLY in this JSON format:
    {{
      "correct": "...",
      "suggestions": ["...", "..."],
      "symptoms": ["...", "..."]
    }}
    """

    try:
        # Generate response
        response = model.generate_content(prompt)
        print("Raw Gemini Response:", response.text)  # Debugging

        # Extract JSON part
        match = re.search(r"\{[\s\S]*\}", response.text)
        if match:
            return json.loads(match.group(0))
        else:
            return {
                "correct": "",
                "suggestions": ["Unable to parse Gemini response."],
                "symptoms": []
            }

    except Exception as e:
        return {
            "correct": "",
            "suggestions": [f"Gemini API error: {str(e)}"],
            "symptoms": []
        }


