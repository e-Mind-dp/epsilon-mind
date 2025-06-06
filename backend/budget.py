# backend/budget.py

from datetime import datetime, timedelta, timezone
from dateutil.parser import isoparse

# DEFAULT_BUDGET = 10.0
# RESET_INTERVAL_HOURS = 24

# Role-based privacy caps
ROLE_BUDGET_CAPS = {
    "researcher": 5.0,
    "doctor": 10.0,
    "public": 3.0
}

REGEN_INTERVAL_MINUTES = 10
REGEN_AMOUNT = 0.5

def get_current_time():
    # return datetime.utcnow()
    # return datetime.now(datetime.timezone.utc)
    return datetime.now(timezone.utc)

def get_role_cap(role: str):
    return ROLE_BUDGET_CAPS.get(role.lower(), 5.0)  # Default to Researcher cap


def should_reset_budget(user_record):
    # last_reset = datetime.fromisoformat(user_record["last_reset"])
    last_reset = isoparse(user_record["last_reset"])
    if last_reset.tzinfo is None:
        last_reset = last_reset.replace(tzinfo=timezone.utc)
    return get_current_time() - last_reset >= timedelta(hours=24)
    # return get_current_time() - last_reset >= timedelta(hours=24)


def regenerate_budget(user_record):
    """
    Regenerate 0.5 budget every 10 minutes up to their role-based cap.
    """
    role = user_record.get("role", "researcher").lower()
    cap = get_role_cap(role)

    # last_updated_str = user_record.get("last_updated", user_record["created_at"])
    # last_updated = datetime.fromisoformat(last_updated_str)
    now = get_current_time()
    # last_updated = datetime.fromisoformat(user_record["last_updated"])
    last_updated = isoparse(user_record["last_updated"])

    if last_updated.tzinfo is None:
        # if for some reason Supabase returns naive datetime, force UTC
        last_updated = last_updated.replace(tzinfo = timezone.utc)

    minutes_elapsed = (now - last_updated).total_seconds() / 60.0
    regen_steps = int(minutes_elapsed // REGEN_INTERVAL_MINUTES)
    regen_total = regen_steps * REGEN_AMOUNT

    if regen_total > 0:
        new_budget = min(cap, user_record["remaining_budget"] + regen_total)
        user_record["remaining_budget"] = new_budget
        user_record["last_updated"] = now.isoformat()

    return user_record


def can_process_query(user_record, epsilon):
    """Check if the user has enough budget, apply reset & regeneration."""
    if should_reset_budget(user_record):
        # Reset to role-specific cap
        role = user_record.get("role", "researcher")
        user_record["remaining_budget"] = get_role_cap(role)
        user_record["last_reset"] = get_current_time().isoformat()
        user_record["last_updated"] = user_record["last_reset"]

    # Regenerate before checking
    user_record = regenerate_budget(user_record)

    if user_record["remaining_budget"] >= epsilon:
        return True, user_record
    else:
        return False, user_record
    

def deduct_budget(user_record, epsilon):
    user_record["remaining_budget"] -= epsilon
    user_record["last_updated"] = get_current_time().isoformat()
    return user_record
