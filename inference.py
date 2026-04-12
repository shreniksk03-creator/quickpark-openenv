import os
import json
from openai import OpenAI
from pydantic import ValidationError

from env import QuickParkEnv, QuickParkAction, ActionType

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct") 
HF_TOKEN = os.getenv("HF_TOKEN") 

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN  
)

def run_baseline():
    print("🚗 Starting Quick Park Admin Simulation...")
    env = QuickParkEnv()
        
    for episode in range(3):
        print(f"\n==============================================")
        print(f"🎬 STARTING EPISODE {episode + 1}")
        print(f"==============================================")
                
        obs = env.reset() 
        max_steps = 3
        
        task_name = f"QuickPark_Task_{episode + 1}"
        
        print(f"[START] task={task_name} env=quickpark model={MODEL_NAME}", flush=True)
                
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
        
        final_score = 0.01
        steps_taken = 0
        rewards_history = [] 
        
        for step in range(max_steps):
            steps_taken += 1
            print(f"\n--- Step {step + 1} ---")
            print(f"👀 AI Observation:\n{obs.model_dump_json(indent=2)}")
            
            action_str = "NOOP"
            error_msg = "null"
            
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
                action_str = action.action_type
                print(f"🤖 AI chose action: {action_str}")
                            
            except Exception as e:
                error_msg = str(e).replace('\n', ' ') 
                print(f"⚠️ Error parsing AI action: {e}. Defaulting to NOOP.")
                action = QuickParkAction(action_type=ActionType.NOOP)
            
            obs, reward, is_done, info = env.step(action)
            
            safe_score = float(reward.score)
            rewards_history.append(f"{safe_score:.2f}")
            final_score = safe_score
                        
            print(f"⚖️ Result: {info['info']}")
            
            done_str = "true" if is_done else "false"
            print(f"[STEP] step={steps_taken} action={action_str} reward={safe_score:.2f} done={done_str} error={error_msg}", flush=True)
                        
            if is_done:
                print(f"\n✅ Task Complete! Reward: {safe_score:.2f}")
                break
        
        success_str = "true" if final_score > 0.5 else "false"
        rewards_csv = ",".join(rewards_history)
        print(f"[END] success={success_str} steps={steps_taken} rewards={rewards_csv}", flush=True)

if __name__ == "__main__":
    run_baseline()
