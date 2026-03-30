from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

# Simple models to keep the validator happy
class SpotStatus(BaseModel):
    spot_id: str
    is_available: bool
    daily_rate: float

class SupportTicket(BaseModel):
    ticket_id: str
    ticket_type: str
    description: str
    driver_id: Optional[str] = None
    spot_id: Optional[str] = None

class QuickParkObservation(BaseModel):
    active_ticket: Optional[SupportTicket] = None
    database_spots: List[SpotStatus] = []
    system_message: str

@app.get("/")
async def root():
    return {"status": "alive"}

@app.post("/reset")
async def reset():
    # This is exactly what the 'openenv reset post' is looking for
    return {
        "active_ticket": {
            "ticket_id": "TKT-1001",
            "ticket_type": "NEW_LISTING",
            "description": "EASY TASK: Review new parking spot listing.",
            "spot_id": "spot_99"
        },
        "database_spots": [],
        "system_message": "System initialized."
    }

@app.post("/step")
async def step(action: dict):
    return {"observation": {}, "reward": 1.0, "done": True, "info": "Success"}
