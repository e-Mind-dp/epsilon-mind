# import streamlit as st
# import requests

# # Set Streamlit page configuration
# st.set_page_config(page_title="SecuQuery", layout="centered")

# # Initialize session state for query history
# if "query_history" not in st.session_state:
#     st.session_state.query_history = []

# # Sidebar: Query History
# with st.sidebar:
#     st.title("Query History")
#     if st.session_state.query_history:
#         for i, entry in enumerate(reversed(st.session_state.query_history), 1):
#             st.markdown(f"**{i}. {entry['query']}**  \n_Answer: {entry['answer']}_", unsafe_allow_html=True)
#     else:
#         st.info("No queries yet!")

# # Main title
# st.markdown("<h1 style='text-align: center;'>SecuQuery</h1>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: center;'>LLM-Powered Query System for Sensitive Data</h5>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>Secure answers powered by Differential Privacy and LLMs</p>", unsafe_allow_html=True)
# st.markdown("---")

# # Layout container
# with st.container():
#     # Dataset selector
#     dataset = st.selectbox("Select Dataset", ("Mental Health", "other"))

#     # Query input
#     query = st.text_input("Enter your query (e.g., What is the average sleep quality?)")

#     # Submit button
#     if st.button('Submit Query'):
#         if query:
#             with st.spinner("Processing your query..."):
#                 payload = {
#                     "query": query,
#                     "dataset": dataset
#                 }

#                 try:
#                     response = requests.post("http://localhost:5000/query", json=payload)
#                     if response.status_code == 200:
#                         result = response.json()

#                         feedback = result.get('feedback', 'No feedback available.')

#                         st.markdown("### Privacy Feedback:")
#                         st.markdown(feedback)
#                         st.session_state.query_history.append({"query": query, "answer": feedback})

#                     elif response.status_code == 400:
#                         result = response.json()
#                         error_message = result.get("error", "Query cannot be processed.")
#                         st.warning(f"⚠️ {error_message}")


#                     else:
#                         st.error(f"Error: Unable to process the request (Code {response.status_code})")

#                 except Exception as e:
#                     st.error(f"Request failed: {e}")
#         else:
#             st.warning("Please enter a query before submitting.")























import streamlit as st
import requests

# Set Streamlit page configuration
st.set_page_config(page_title="SecuQuery", layout="centered")

BASE_URL = "http://localhost:5000"

# Session state initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "user_query_histories" not in st.session_state:
    st.session_state.user_query_histories = {}

# ------------------------------
# MAIN: Login/Register Screen
# ------------------------------
if st.session_state.user_id is None:
    st.title("🔐 Welcome to SecuQuery")
    st.subheader("Please login or register to continue")

    auth_option = st.radio("Choose an option", ["Login", "Register"], horizontal=True)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if auth_option == "Register":
        role = st.selectbox("Select your role", ["Researcher", "Doctor", "Public"])
        if st.button("Register"):
            res = requests.post(f"{BASE_URL}/register", json={"email": email, "password": password, "role": role})
            if res.status_code == 201:
                st.success("✅ Registered successfully! Please login.")
            else:
                st.error(res.json().get("error", "❌ Registration failed."))

    elif auth_option == "Login":
        if st.button("Login"):
            res = requests.post(f"{BASE_URL}/login", json={"email": email, "password": password})
            if res.status_code == 200:
                st.success("✅ Login successful.")
                st.session_state.user_id = res.json()["user_id"]

                # Initialize query history for this user if not exists
                if st.session_state.user_id not in st.session_state.user_query_histories:
                    st.session_state.user_query_histories[st.session_state.user_id] = []

                st.rerun()
                st.stop() 
            else:
                st.error(res.json().get("error", "❌ Login failed."))

# ------------------------------
# MAIN: Query UI (after login)
# ------------------------------
else:

    # Ensure user history is available
    if st.session_state.user_id not in st.session_state.user_query_histories:
        st.session_state.user_query_histories[st.session_state.user_id] = []

    history = st.session_state.user_query_histories[st.session_state.user_id]


    # Sidebar with logout and history
    with st.sidebar:
        st.success("✅ Logged in")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.rerun()
            st.stop()

        st.title("📜 Query History")
        # if st.session_state.query_history:
        if history:
            # for i, entry in enumerate(reversed(st.session_state.query_history), 1):
            for i, entry in enumerate(reversed(history), 1):
                st.markdown(f"**{i}. {entry['query']}**  \n_Answer: {entry['answer']}_", unsafe_allow_html=True)
        else:
            st.info("No queries yet!")

    # Main title
    st.markdown("<h1 style='text-align: center;'>SecuQuery</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>LLM-Powered Query System for Sensitive Data</h5>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>Secure answers powered by Differential Privacy and LLMs</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Dataset selector
    dataset = st.selectbox("📊 Select Dataset", ("Mental Health", "Other"))

    # Query input
    query = st.text_input("💬 Enter your query (e.g., What is the average sleep quality?)")

    # Submit button
    if st.button("Submit Query"):
        if query:
            with st.spinner("🔐 Processing your query securely..."):
                payload = {
                    "query": query,
                    "dataset": dataset,
                    "user_id": st.session_state.user_id
                }

                try:
                    response = requests.post(f"{BASE_URL}/query", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        feedback = result.get("feedback", "No feedback available.")
                        st.markdown("### 🛡️ Privacy Feedback:")
                        st.markdown(feedback)
                        # st.session_state.query_history.append({"query": query, "answer": feedback})
                        history.append({"query": query, "answer": feedback})
                        
                    elif response.status_code == 400:
                        error_message = response.json().get("error", "Query cannot be processed.")
                        st.warning(f"⚠️ {error_message}")
                    else:
                        st.error(f"Error: Unable to process the request (Code {response.status_code})")
                except Exception as e:
                    st.error(f"Request failed: {e}")
        else:
            st.warning("Please enter a query before submitting.")
