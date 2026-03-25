"""Image parser using GPT-4V for description generation."""
import base64, os
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ImageParser:
    def describe(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":[
                {"type":"text","text":"Describe this image in detail for use in a document search system. Include all text, numbers, chart data, and visual elements."},
                {"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}}
            ]}],
            max_tokens=800
        )
        return response.choices[0].message.content
