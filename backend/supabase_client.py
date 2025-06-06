from supabase import create_client, Client

url = "https://inooaduglrsezmfdkbtg.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlub29hZHVnbHJzZXptZmRrYnRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY2MzYxMTIsImV4cCI6MjA2MjIxMjExMn0.lbjC4JtHxcctPrATz9ahpzK47K_LLI488KKOCGB9Ei4"

supabase: Client = create_client(url, key)
