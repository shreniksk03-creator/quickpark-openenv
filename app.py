from fastapi import FastAPI
from env import QuickParkEnv, QuickParkAction

app = FastAPI()
simulation = QuickParkEnv()

@app.get("/")
def health_check():
    """The automated ping hits here to check if it returns 200 OK."""
    return {"status": "200 OK", "message": "QuickPark Environment is running."}

@app.post("/reset")
def reset_env():
    return simulation.reset()

@app.post("/step")
def step_env(action: QuickParkAction):
    obs, reward, done, info = simulation.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
def get_state():
    return simulation.state()