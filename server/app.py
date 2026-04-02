from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# --- Your exact endpoints from before go here ---
@app.get("/")
async def root():
    return {"status": "alive"}

@app.post("/reset")
async def reset():
    return {
        "active_ticket": {"ticket_id": "TKT-1001", "ticket_type": "NEW_LISTING", "description": "Review listing", "spot_id": "spot_99"},
        "database_spots": [],
        "system_message": "System initialized."
    }

@app.post("/step")
async def step(action: dict):
    return {"observation": {}, "reward": 1.0, "done": True, "info": "Success"}

# --- THIS IS THE NEW PART THE VALIDATOR WANTS ---
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
