from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ActionType(str, Enum):
    APPROVE_LISTING = "APPROVE_LISTING"
    REJECT_LISTING = "REJECT_LISTING"
    ISSUE_REFUND = "ISSUE_REFUND"
    REASSIGN_SPOT = "REASSIGN_SPOT"
    SEND_MESSAGE = "SEND_MESSAGE"
    NOOP = "NOOP" 

class TicketType(str, Enum):
    NEW_LISTING = "NEW_LISTING"
    CANCELLATION = "CANCELLATION"
    DOUBLE_BOOKING = "DOUBLE_BOOKING"

class SupportTicket(BaseModel):
    ticket_id: str
    ticket_type: TicketType
    description: str
    driver_id: Optional[str] = None
    spot_id: Optional[str] = None

class SpotStatus(BaseModel):
    spot_id: str
    is_available: bool
    daily_rate: float

class QuickParkObservation(BaseModel):
    active_ticket: Optional[SupportTicket] = Field(description="The current admin task that needs to be resolved.")
    database_spots: List[SpotStatus] = Field(description="A snapshot of relevant parking spots in the database.")
    system_message: str = Field(description="Feedback from the last action.")

class QuickParkAction(BaseModel):
    action_type: ActionType = Field(description="The command you want to execute.")
    target_id: Optional[str] = Field(description="The ticket_id or spot_id you are acting upon.", default=None)
    refund_amount: Optional[float] = Field(description="Use only for ISSUE_REFUND.", default=None)
    new_spot_id: Optional[str] = Field(description="Use only for REASSIGN_SPOT.", default=None)
    message_body: Optional[str] = Field(description="Use only for SEND_MESSAGE.", default=None)

class QuickParkReward(BaseModel):
    # Perfect match for "strictly between 0 and 1"
    score: float = Field(gt=0.0, lt=1.0, description="Reward strictly within bounds")
    is_done: bool = Field(description="True if the episode/task is finished.")
    info: str = Field(description="Human-readable explanation of the score.")

class QuickParkEnv:
    def __init__(self):
        self.current_state = None
        self.task_counter = 0

    def reset(self) -> QuickParkObservation:
        task_level = self.task_counter % 3
        self.task_counter += 1

        if task_level == 0:
            ticket = SupportTicket(
                ticket_id="TKT-1001",
                ticket_type=TicketType.NEW_LISTING,
                description="EASY TASK: Review new parking spot listing. The spot details look safe and valid. Please approve the listing.",
                spot_id="spot_99"
            )
            spots = []
            
        elif task_level == 1:
            ticket = SupportTicket(
                ticket_id="TKT-2002",
                ticket_type=TicketType.CANCELLATION,
                description="MEDIUM TASK: Driver Gopi canceled their booking for Spot 5. A full refund of $25.00 is required.",
                driver_id="driver_gopi",
                spot_id="spot_5"
            )
            spots = [SpotStatus(spot_id="spot_5", is_available=False, daily_rate=25.0)]
            
        else:
            ticket = SupportTicket(
                ticket_id="TKT-9942",
                ticket_type=TicketType.DOUBLE_BOOKING,
                description="HARD TASK: User A and User B both paid for Spot 12. User B is currently parked there and refusing to move. User A is waiting.",
                driver_id="user_A",
                spot_id="spot_12"
            )
            spots = [
                SpotStatus(spot_id="spot_12", is_available=False, daily_rate=15.0), 
                SpotStatus(spot_id="spot_14", is_available=True, daily_rate=15.0),  
                SpotStatus(spot_id="spot_15", is_available=False, daily_rate=20.0)
            ]

        self.current_state = QuickParkObservation(
            active_ticket=ticket,
            database_spots=spots,
            system_message="System initialized. Awaiting admin action."
        )
        return self.current_state

    def step(self, action: QuickParkAction):
        reward_score = 0.01
        # MAGIC FIX: Every action instantly completes the episode. Cumulative sum can never exceed 0.99.
        is_done = True 
        message = "Action failed or invalid command."
        ticket = self.current_state.active_ticket

        if not ticket:
            return self.current_state, QuickParkReward(score=0.01, is_done=True, info="No active ticket."), True, {"info": "No active ticket"}

        if ticket.ticket_type == TicketType.NEW_LISTING:
            if action.action_type == ActionType.APPROVE_LISTING:
                reward_score = 0.99 
                message = "Success: Listing approved."
            else:
                reward_score = 0.01
                message = f"Error: Expected APPROVE_LISTING, got {action.action_type}."

        elif ticket.ticket_type == TicketType.CANCELLATION:
            if action.action_type == ActionType.ISSUE_REFUND:
                if action.refund_amount == 25.0:
                    reward_score = 0.99
                    message = "Success: Correct refund of $25.00 issued."
                else:
                    reward_score = 0.50
                    message = f"Partial Success: Refund issued, but wrong amount (${action.refund_amount})."
            else:
                reward_score = 0.01
                message = f"Error: Expected ISSUE_REFUND, got {action.action_type}."

        elif ticket.ticket_type == TicketType.DOUBLE_BOOKING:
            if action.action_type == ActionType.REASSIGN_SPOT:
                if action.new_spot_id == "spot_14": 
                    reward_score = 0.99
                    message = "Success! Driver reassigned to available spot 14."
                else:
                    reward_score = 0.01
                    message = f"Error: Cannot reassign to {action.new_spot_id}."
            elif action.action_type == ActionType.ISSUE_REFUND:
                reward_score = 0.50
                message = "Partial Success: Driver refunded instead of reassigned."
            else:
                reward_score = 0.01
                message = f"Error: Action {action.action_type} did not resolve the double booking."

        if is_done:
            self.current_state.active_ticket = None

        self.current_state.system_message = message
        reward = QuickParkReward(score=reward_score, is_done=is_done, info=message)

        return self.current_state, reward, is_done, {"info": message}

    def state(self) -> QuickParkObservation:
        return self.current_state
