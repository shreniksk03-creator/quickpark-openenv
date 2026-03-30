import os
import json
from openai import OpenAI
from pydantic import ValidationError

# Import our simulation from the env.py file we just made!
from env import QuickParkEnv, QuickParkAction, ActionType

# --- 1. SETUP API CREDENTIALS ---
# The hackathon strictly requires these exact variable names
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# We will pass the token in the terminal, but leave a placeholder just in case
API_KEY = os.getenv("HF_TOKEN", "YOUR_TOKEN_HERE") 
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct") 

# Initialize the OpenAI client pointing to Hugging Face
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def run_baseline():
    print("🚗 Starting Quick Park Admin Simulation...")
    env = QuickParkEnv()
    
    # We loop 3 times to play all 3 tasks (Easy, Medium, Hard)
    for episode in range(3):
        print(f"\n==============================================")
        print(f"🎬 STARTING EPISODE {episode + 1}")
        print(f"==============================================")
        
        obs = env.reset() 
        max_steps = 3
        
        system_prompt = """
        You are an expert AI admin for the Quick Park app.
        Your goal is to resolve the active support ticket.
        
        CRITICAL RULES:
        1. Read the 'database_spots' carefully. If a spot's 'is_available' is true, you can use it.
        2. If you use REASSIGN_SPOT, you MUST include the exact 'new_spot_id' from the database.
        3. If you use ISSUE_REFUND, you MUST include the 'refund_amount'.
        
        You must respond with ONLY a raw JSON object matching this schema, nothing else:
        {
            "action_type": "REASSIGN_SPOT" | "ISSUE_REFUND" | "APPROVE_LISTING" | "REJECT_LISTING" | "SEND_MESSAGE" | "NOOP",
            "target_id": "string (optional)",
            "refund_amount": float (optional),
            "new_spot_id": "string (optional)",
            "message_body": "string (optional)"
        }
        """

        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            print(f"👀 AI Observation:\n{obs.model_dump_json(indent=2)}")
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Current State: {obs.model_dump_json()}"}
                    ],
                    temperature=0.1 
                )
                
                ai_response = response.choices[0].message.content.strip()
                if ai_response.startswith("```json"):
                    ai_response = ai_response[7:-3].strip()
                elif ai_response.startswith("```"):
                    ai_response = ai_response[3:-3].strip()
                    
                action_dict = json.loads(ai_response)
                action = QuickParkAction(**action_dict)
                print(f"🤖 AI chose action: {action.action_type}")
                
            except Exception as e:
                print(f"⚠️ Error parsing AI action: {e}. Defaulting to NOOP.")
                action = QuickParkAction(action_type=ActionType.NOOP)

            obs, reward, is_done, info = env.step(action)
            
            print(f"⚖️ Result: {info['info']}")
            print(f"🏆 Score: {reward.score}")
            
            if is_done:
                print(f"\n✅ Task Complete! Final Score: {reward.score} / 1.0")
                break

if __name__ == "__main__":
    run_baseline()