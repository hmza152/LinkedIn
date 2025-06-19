import os
import requests
import urllib.parse
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from dotenv import load_dotenv
import uvicorn  # Import uvicorn to run the app

# Load environment variables from .env file
load_dotenv()

# --- LinkedIn App Credentials (loaded from .env) ---
# It's crucial to load these from environment variables and NOT hardcode them.
CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
# This must EXACTLY match one of the Redirect URLs configured in your LinkedIn app
# Ensure this matches the port your FastAPI app will run on (default 8000 for uvicorn)
REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")

# --- 2. Define LinkedIn OAuth Endpoints ---
AUTHORIZATION_BASE_URL = "https://www.linkedin.com/oauth/v2/authorization"
ACCESS_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
USER_INFO_URL = (
    "https://api.linkedin.com/v2/userinfo"  # For OpenID Connect profile data
)

# --- 3. Scopes (Permissions) Your App Needs ---
# Common scopes:
# openid: Required for OpenID Connect, provides a unique user ID
# profile: Grants access to basic profile info (name, photo, headline)
# email: Grants access to the user's primary email address
SCOPES = [
    "openid",
    "profile",
    "email",
]  # Add more as needed, e.g., "w_member_social" for sharing


# --- STEP 1: Generate the Authorization URL ---
# This is the URL your user will be redirected to (e.g., by clicking a "Sign in with LinkedIn" button)
def get_linkedin_authorization_url():
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),  # Scopes are space-separated
        "state": "random_string_for_security",  # Recommended for CSRF protection
    }
    auth_url = f"{AUTHORIZATION_BASE_URL}?{urllib.parse.urlencode(params)}"
    return auth_url


# --- STEP 2: Handle the Callback and Exchange Code for Access Token ---
# This function would be called on your server-side when LinkedIn redirects the user
# back to your REDIRECT_URI with the authorization code.
def exchange_code_for_token(authorization_code, state_received):
    # IMPORTANT: In a real app, you'd compare state_received with the state you sent
    # to prevent CSRF attacks.

    token_params = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    try:
        response = requests.post(ACCESS_TOKEN_URL, data=token_params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        token_data = response.json()
        return token_data
    except requests.exceptions.RequestException as e:
        print(f"Error exchanging code for token: {e}")
        return None


# --- STEP 3: Use the Access Token to Fetch User Data ---
def get_user_profile(access_token):
    headers = {"Authorization": f"Bearer {access_token}", "Connection": "Keep-Alive"}
    try:
        # For basic OpenID Connect profile info
        response = requests.get(USER_INFO_URL, headers=headers)
        response.raise_for_status()
        user_info = response.json()
        return user_info
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user profile: {e}")
        return None


# --- Demonstration Flow ---
if __name__ == "__main__":
    print("--- LinkedIn OAuth 2.0 Flow ---")

    # Step 1: Get the Authorization URL
    auth_url = get_linkedin_authorization_url()
    print(f"\n1. Please open this URL in your browser to authorize your app:")
    print(auth_url)
    print(
        "\nAfter authorization, LinkedIn will redirect you to your REDIRECT_URI (e.g., http://localhost:5000/callback)."
    )
    print(
        "You'll need to manually copy the 'code' parameter from the URL in your browser's address bar."
    )

    # --- SIMULATE THE REDIRECT AND CODE EXTRACTION ---
    # In a real web app, your /callback route would extract this 'code'
    # from the incoming request's query parameters.
    print(
        "\n2. Paste the 'code' parameter from the redirected URL here (e.g., code=AQQ...):"
    )
    authorization_code = input("Enter the authorization code: ").strip()
    state_received_from_linkedin = input(
        "Enter the 'state' parameter (if present, for verification): "
    ).strip()

    if not authorization_code:
        print("No authorization code entered. Exiting.")
    else:
        # Step 2: Exchange the code for an Access Token
        print("\n3. Exchanging authorization code for access token...")
        token_data = exchange_code_for_token(
            authorization_code, state_received_from_linkedin
        )

        if token_data:
            print("\nAccess Token Data:")
            print(json.dumps(token_data, indent=2))
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in")
            # You might also get a 'refresh_token' depending on scopes and app settings

            if access_token:
                print(f"\nAccess Token obtained! Valid for {expires_in} seconds.")

                # Step 3: Use the Access Token to fetch user data
                print("\n4. Fetching user profile using the access token...")
                user_profile = get_user_profile(access_token)
                if user_profile:
                    print("\nUser Profile Data:")
                    print(json.dumps(user_profile, indent=2))
                else:
                    print("Failed to fetch user profile.")
            else:
                print("Access token not found in the response.")
        else:
            print("Failed to get access token.")
