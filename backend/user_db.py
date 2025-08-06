import bcrypt
from supabase_client import supabase
from datetime import datetime, timedelta
import uuid
from budget import get_role_cap, should_reset_budget


def get_user_by_email(email):
    """Fetch user from Supabase by email."""
    try:
        response = supabase.table("users").select("*").eq("email", email).execute()
        if response.data:
            return response.data[0]  
        return None  
    except Exception as e:
        print("Error fetching user:", e)
        return None



def hash_password(password):
    """Hash a plain-text password."""
    salt = bcrypt.gensalt()  
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)  
    return hashed_password.decode('utf-8') 



def register_user(email, password, role="Unauthorised"):
    cap = get_role_cap(role)

    try:
        hashed_password = hash_password(password) 
        user_id = str(uuid.uuid4()) 
        response = supabase.table("users").insert({
            "id": user_id,
            "email": email,
            "password": hashed_password,
            "role": role,
            "privacy_budget": cap,
            "remaining_budget": cap,
            "created_at": datetime.utcnow().isoformat(),
            "last_reset": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat() 
        }).execute()
        return response.data
    except Exception as e:
        print("Error registering user:", e)
        return None
    


def get_user_record(user_id):
    result = supabase.table("users").select("*").eq("id", user_id).execute()
    users = result.data
    if not users:
        return None

    user = users[0]

    # Reset if needed using shared function
    if should_reset_budget(user):
        cap = get_role_cap(user["role"])
        user["remaining_budget"] = cap
        user["last_reset"] = datetime.utcnow().isoformat()
        user["last_updated"] = user["last_reset"]
        update_user_record(user_id, user)

    return user


def update_user_record(user_id, updated_record):
    try:
        supabase.table("users").update({
            "remaining_budget": updated_record["remaining_budget"],
            "last_reset": updated_record["last_reset"],
            "last_updated": updated_record["last_updated"]
        }).eq("id", user_id).execute()
    except Exception as e:
        print("Update error:", e)



def store_user_query(user_id, query_text, epsilon_used, sensitivity):
    try:
        query_id = str(uuid.uuid4())
        supabase.table("user_queries").insert({
            "id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "epsilon_used": epsilon_used,
            "sensitivity": sensitivity,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
        print("Query stored successfully")
    except Exception as e:
        print("Error storing query:", e)


def get_all_user_queries(user_id):
    """Fetch all queries previously asked by a user."""
    try:
        response = supabase.table("user_queries") \
            .select("query_text") \
            .eq("user_id", user_id) \
            .order("timestamp", desc=False) \
            .execute()

        if response.data:
            return [entry["query_text"] for entry in response.data]
        return []
    except Exception as e:
        print("Error fetching user queries:", e)
        return []

