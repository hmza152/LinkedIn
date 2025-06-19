import os
import requests
import urllib.parse
import json
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import RedirectResponse, JSONResponse
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel, HttpUrl
from typing import Annotated, Optional, List
from bs4 import BeautifulSoup
import openai
from datetime import datetime, timedelta
import pytz
import re
from openai import OpenAI
import json
from typing import Optional
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI(
    title="LinkedIn API Integration with FastAPI",
    description="A comprehensive FastAPI application for LinkedIn OAuth 2.0, handling user authentication, token management, and posting text, articles, and images.",
    version="1.2.0",
)


# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Credentials ---
CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directories
TOKEN_STORAGE_DIR = "user_tokens"
SCHEDULED_POSTS_FILE = "scheduled_posts.json"
os.makedirs(TOKEN_STORAGE_DIR, exist_ok=True)

# Credential validation
if not all([CLIENT_ID, CLIENT_SECRET, REDIRECT_URI]):
    raise ValueError(
        "Missing LinkedIn API credentials. Set LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET, and LINKEDIN_REDIRECT_URI in .env."
    )
if not OPENAI_API_KEY:
    print(
        "Warning: OPENAI_API_KEY missing. Post generation may fail if no text provided."
    )

# --- LinkedIn API Endpoints ---
AUTHORIZATION_BASE_URL = "https://www.linkedin.com/oauth/v2/authorization"
ACCESS_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
USER_INFO_URL = "https://api.linkedin.com/v2/userinfo"
UGC_POSTS_API_URL = "https://api.linkedin.com/v2/ugcPosts"
POSTS_API_URL = "https://api.linkedin.com/v2/posts"
ASSET_API_URL = "https://api.linkedin.com/v2/assets?action=registerUpload"

# --- Scopes ---
SCOPES = ["openid", "profile", "email", "w_member_social"]

# CSRF state storage
states = {}


# --- Pydantic Models ---
class PostContent(BaseModel):
    text: str


class ArticleContent(BaseModel):
    text: str
    url: HttpUrl


class ImageContent(BaseModel):
    text: str
    image_urn: str


class ScheduledPost(BaseModel):
    text: str
    image_path: Optional[str] = None
    url: Optional[HttpUrl] = None
    timestamp: str


# --- Helper Functions ---
def save_user_credentials(user_id: str, credentials: dict):
    file_path = os.path.join(TOKEN_STORAGE_DIR, f"{user_id}.json")
    try:
        with open(file_path, "w") as f:
            json.dump(credentials, f, indent=4)
        print(f"Credentials for user {user_id} saved to {file_path}")
    except IOError as e:
        print(f"Error saving credentials for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save user credentials: {e}"
        )


def load_user_credentials(user_id: str) -> dict | None:
    file_path = os.path.join(TOKEN_STORAGE_DIR, f"{user_id}.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading credentials for user {user_id}: {e}")
        return None


def get_linkedin_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": "202405",
    }


def handle_linkedin_api_error(e: requests.exceptions.RequestException):
    print(f"Error communicating with LinkedIn API: {e}")
    status_code = 500
    linkedin_error_details = {"message": "An unexpected error occurred."}
    if e.response is not None:
        status_code = e.response.status_code
        try:
            linkedin_error_details = e.response.json()
            print(
                f"LinkedIn API error details: {json.dumps(linkedin_error_details, indent=2)}"
            )
        except json.JSONDecodeError:
            linkedin_error_details = {"message": e.response.text or "No response text."}
    raise HTTPException(
        status_code=status_code,
        detail={
            "message": "Failed to communicate with LinkedIn.",
            "linkedin_api_response": linkedin_error_details,
        },
    )


def save_scheduled_posts(posts: List[dict]):
    try:
        with open(SCHEDULED_POSTS_FILE, "w") as f:
            json.dump(posts, f, indent=4)
    except IOError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save scheduled posts: {e}"
        )


def load_scheduled_posts() -> List[dict]:
    if not os.path.exists(SCHEDULED_POSTS_FILE):
        return []
    try:
        with open(SCHEDULED_POSTS_FILE, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading scheduled posts: {e}")
        return []


def parse_user_prompt(
    prompt: str,
) -> tuple[int, Optional[str], Optional[str], Optional[str]]:
    """Parse user prompt to extract number of posts, schedule, description, and post text."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key missing.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only parser. "
                        "Parse a prompt related to scheduling LinkedIn posts. "
                        "Return this exact JSON format:\n"
                        "{"
                        '"num_posts": int, '
                        '"schedule": str or null, '
                        '"description": str or null, '
                        '"post_text": str or null'
                        "}\n"
                        "Do not include any explanation, just the JSON."
                    ),
                },
                {"role": "user", "content": f"Parse this prompt: '{prompt}'"},
            ],
            max_tokens=300,
            temperature=0.5,
        )

        content = response.choices[0].message.content.strip()
        print(f"LLM raw response:\n{content}")

        result = json.loads(content)
        num_posts = int(result.get("num_posts", 1))
        schedule = result.get("schedule")
        description = result.get("description")
        post_text = result.get("post_text")

        print(
            f"Parsed prompt: num_posts={num_posts}, schedule={schedule}, description={description}, post_text={post_text}"
        )

        return num_posts, schedule, description, post_text

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON returned by OpenAI: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error parsing prompt with OpenAI: {e}"
        )


def generate_post_content(description: str, link: Optional[HttpUrl] = None) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key missing.")
    openai.api_key = OPENAI_API_KEY
    content = description
    if link:
        try:
            response = requests.get(link)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text(separator=" ", strip=True)[:1000]
            print("BeautifulSoup content fetched successfully.", content[:100])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching content from {link}: {e}")
            content = description
    try:
        print("starting CHATGPT")
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Create a LinkedIn post based on this: {content}. Include a CTA and 1-3 hashtags.",
                },
            ],
            temperature=0.7,
        )
        print("CHATGPT response received", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating post with OpenAI: {e}"
        )


def parse_schedule_to_utc(schedule: str, start_date: datetime) -> List[str]:
    time_match = re.search(
        r"(\d{1,2}(?::\d{2})?\s*(AM|PM))\s*(EST|EDT)", schedule, re.IGNORECASE
    )
    if not time_match:
        raise HTTPException(
            status_code=400,
            detail="Invalid schedule format. Use 'daily at HH:MM AM/PM EST'.",
        )
    time_str, am_pm, tz = time_match.groups()
    est = pytz.timezone("US/Eastern")
    time_format = "%I:%M %p" if ":" in time_str else "%I %p"
    time_obj = datetime.strptime(time_str, time_format).time()
    est_datetime = est.localize(datetime.combine(start_date.date(), time_obj))
    utc_datetime = est_datetime.astimezone(pytz.UTC)
    timestamps = []
    for i in range(10):
        daily_utc = utc_datetime + timedelta(days=i)
        timestamps.append(daily_utc.isoformat())
    return timestamps


# --- OAuth Endpoints ---
@app.get("/", summary="Get LinkedIn Login URL")
async def get_login_url():
    state = os.urandom(16).hex()
    states[state] = True
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "state": state,
    }
    auth_url = f"{AUTHORIZATION_BASE_URL}?{urllib.parse.urlencode(params)}"
    return JSONResponse(content={"linkedin_login_url": auth_url})


@app.get("/getID", summary="LinkedIn OAuth Callback - Return user_id")
async def linkedin_id(code: str, state: str):
    # Validate state to prevent CSRF attacks
    if state not in states:
        raise HTTPException(status_code=403, detail="Invalid or expired state.")
    states.pop(state, None)

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }

    try:
        # Exchange code for access token
        token_response = requests.post(
            ACCESS_TOKEN_URL,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_response.raise_for_status()
        token_json = token_response.json()
        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve access token"
            )

        # Fetch user info
        headers = {"Authorization": f"Bearer {access_token}"}
        profile_response = requests.get(USER_INFO_URL, headers=headers)
        profile_response.raise_for_status()
        profile = profile_response.json()

        user_id = profile.get("sub")  # Use 'sub' as per OpenID Connect standard
        if not user_id:
            raise HTTPException(status_code=500, detail="Failed to retrieve user ID")

        # Save credentials
        credentials_to_save = {"user_profile": profile, **token_json}
        save_user_credentials(user_id, credentials_to_save)

        return {"user_id": user_id, "profile": profile}

    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)


@app.get("/callback", summary="LinkedIn OAuth Callback")
async def linkedin_callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    if error:
        return RedirectResponse(url=f"/auth/error?error={error}")
    if not state or state not in states:
        raise HTTPException(
            status_code=403, detail="State mismatch. Possible CSRF attack."
        )
    del states[state]
    token_params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    try:
        response = requests.post(ACCESS_TOKEN_URL, data=token_params)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        headers = {"Authorization": f"Bearer {access_token}"}
        user_info_response = requests.get(USER_INFO_URL, headers=headers)
        user_info_response.raise_for_status()
        user_profile = user_info_response.json()
        user_id = user_profile.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=500, detail="Could not retrieve user ID from profile."
            )
        print(f"User authenticated! LinkedIn User ID: {user_id}")
        credentials_to_save = {"user_profile": user_profile, **token_data}
        save_user_credentials(user_id, credentials_to_save)
        return RedirectResponse(
            url=f"https://overos.xyz/linkedin/callback?user_id={user_id}"
        )
    except requests.exceptions.RequestException as e:
        return handle_linkedin_api_error(e)


@app.get("/auth/success", summary="LinkedIn Authentication Success")
async def auth_success(user_id: str):
    return JSONResponse(
        content={"message": "LinkedIn authentication successful!", "user_id": user_id}
    )


@app.get("/auth/error", summary="LinkedIn Authentication Error")
async def auth_error(error: str = "unknown_error"):
    return JSONResponse(
        status_code=400,
        content={"message": "LinkedIn authentication failed.", "error": error},
    )


# --- LinkedIn Share API Endpoints ---
def get_and_validate_creds(user_id: str):
    credentials = load_user_credentials(user_id)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"Credentials not found for user {user_id}. Please log in first.",
        )
    scope_string = credentials.get("scope", "")
    scopes = scope_string.replace(",", " ").split()
    if "w_member_social" not in scopes:
        raise HTTPException(
            status_code=403,
            detail="The 'w_member_social' scope is required for posting.",
        )
    return credentials


@app.post("/users/{user_id}/share_text", summary="Post a Text Update to LinkedIn")
async def share_text_post(user_id: str, content: PostContent):
    creds = get_and_validate_creds(user_id)
    post_payload = {
        "author": f"urn:li:person:{user_id}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": content.text},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    try:
        response = requests.post(
            UGC_POSTS_API_URL,
            headers=get_linkedin_headers(creds["access_token"]),
            json=post_payload,
        )
        response.raise_for_status()
        post_urn = response.headers.get("x-restli-id", "")
        return JSONResponse(
            status_code=response.status_code,
            content={"message": "Text post shared successfully!", "post_urn": post_urn},
        )
    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)


@app.post("/users/{user_id}/share_article", summary="Share an Article/URL on LinkedIn")
async def share_article_post(user_id: str, content: ArticleContent):
    creds = get_and_validate_creds(user_id)
    post_payload = {
        "author": f"urn:li:person:{user_id}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": content.text},
                "shareMediaCategory": "ARTICLE",
                "media": [{"status": "READY", "originalUrl": str(content.url)}],
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    try:
        response = requests.post(
            UGC_POSTS_API_URL,
            headers=get_linkedin_headers(creds["access_token"]),
            json=post_payload,
        )
        response.raise_for_status()
        post_urn = response.headers.get("x-restli-id", "")
        return JSONResponse(
            status_code=response.status_code,
            content={"message": "Article shared successfully!", "post_urn": post_urn},
        )
    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)


@app.post(
    "/users/{user_id}/register_image_upload",
    summary="Step 1: Register an Image for Upload",
)
async def register_image_upload(user_id: str):
    creds = get_and_validate_creds(user_id)
    register_payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": f"urn:li:person:{user_id}",
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }
            ],
        }
    }
    try:
        response = requests.post(
            ASSET_API_URL,
            headers=get_linkedin_headers(creds["access_token"]),
            json=register_payload,
        )
        response.raise_for_status()
        upload_info = response.json().get("value", {})
        asset_urn = upload_info.get("asset")
        upload_url = (
            upload_info.get("uploadMechanism", {})
            .get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {})
            .get("uploadUrl")
        )
        if not all([asset_urn, upload_url]):
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve valid upload URL or asset URN.",
            )
        return JSONResponse(content={"asset_urn": asset_urn, "upload_url": upload_url})
    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)


@app.post("/users/{user_id}/share_image", summary="Step 2: Share an Image Post")
async def share_image_post(user_id: str, content: ImageContent):
    creds = get_and_validate_creds(user_id)
    post_payload = {
        "author": f"urn:li:person:{user_id}",
        "lifecycleState": "PUBLIC",
        "content": {
            "contentEntities": [{"entity": content.image_urn, "thumbnails": []}],
            "description": content.text,
            "shareMediaCategory": "IMAGE",
        },
        "visibility": {"PUBLIC": "anyone"},
    }
    try:
        response = requests.post(
            POSTS_API_URL,
            headers=get_linkedin_headers(creds["access_token"]),
            json=post_payload,
        )
        response.raise_for_status()
        post_urn = response.headers.get("x-restli-id", "")
        return JSONResponse(
            status_code=response.status_code,
            content={
                "message": "Image post shared successfully!",
                "post_urn": post_urn,
            },
        )
    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)


@app.post(
    "/users/{user_id}/upload_and_share_image",
    summary="Upload Raw Image and Share with Commentary",
)
async def upload_and_share_image(
    user_id: str,
    image_file: Annotated[
        UploadFile, File(description="The raw image file to upload (JPEG, PNG, GIF)")
    ],
    text: Annotated[str, Form(description="Commentary text for the LinkedIn post")],
    url: Annotated[
        Optional[HttpUrl],
        Form(description="Optional: External URL to associate with the image post"),
    ] = None,
):
    creds = get_and_validate_creds(user_id)
    access_token = creds["access_token"]
    register_payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": f"urn:li:person:{user_id}",
            "serviceRelationships": [
                {
                    "relationshipType": "OWNER",
                    "identifier": "urn:li:userGeneratedContent",
                }
            ],
        }
    }
    try:
        register_response = requests.post(
            ASSET_API_URL,
            headers=get_linkedin_headers(access_token),
            json=register_payload,
        )
        register_response.raise_for_status()
        upload_info = register_response.json().get("value", {})
        asset_urn = upload_info.get("asset")
        upload_url = (
            upload_info.get("uploadMechanism", {})
            .get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {})
            .get("uploadUrl")
        )
        if not all([asset_urn, upload_url]):
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve valid upload URL or asset URN.",
            )
        image_data = await image_file.read()
        upload_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/octet-stream",
        }
        upload_response = requests.put(
            upload_url, headers=upload_headers, data=image_data
        )
        upload_response.raise_for_status()
        media_content = {"status": "READY", "media": asset_urn}
        if url:
            media_content["originalUrl"] = str(url)
        post_payload = {
            "author": f"urn:li:person:{user_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": text},
                    "shareMediaCategory": "IMAGE",
                    "media": [media_content],
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        share_response = requests.post(
            UGC_POSTS_API_URL,
            headers=get_linkedin_headers(access_token),
            json=post_payload,
        )
        share_response.raise_for_status()
        post_urn = share_response.headers.get("x-restli-id", "")
        return JSONResponse(
            status_code=share_response.status_code,
            content={
                "message": "Image post shared successfully!",
                "post_urn": post_urn,
                "asset_urn": asset_urn,
            },
        )
    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/users/{user_id}/universal_post", summary="Universal Post to LinkedIn")
async def universal_post(
    user_id: str,
    text: Annotated[str, Form(description="Commentary text for the LinkedIn post")],
    image_file: Annotated[
        Optional[UploadFile],
        File(description="Optional: The raw image file to upload (JPEG, PNG, GIF)"),
    ] = None,
    url: Annotated[
        Optional[HttpUrl],
        Form(description="Optional: External URL to associate with the post"),
    ] = None,
):
    creds = get_and_validate_creds(user_id)
    try:
        if image_file:
            return await upload_and_share_image(
                user_id=user_id, image_file=image_file, text=text, url=url
            )
        elif url:
            content = ArticleContent(text=text, url=url)
            return await share_article_post(user_id=user_id, content=content)
        else:
            content = PostContent(text=text)
            return await share_text_post(user_id=user_id, content=content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing the post: {e}",
        )


@app.post("/users/{user_id}/smart_post", summary="Smart Post to LinkedIn")
async def smart_post(
    user_id: str,
    user_prompt: Annotated[
        str,
        Form(
            description="Prompt describing the post(s) and optional schedule (e.g., 'Post this: Hello world! or Create 10 posts for my startup at https://myaisaas.com every day at 9 AM EST')"
        ),
    ],
    image_files: Annotated[
        Optional[List[UploadFile]],
        File(description="Optional: List of image files for the posts"),
    ] = None,
    urls: Annotated[
        Optional[List[HttpUrl]],
        Form(description="Optional: List of URLs to associate with the posts"),
    ] = None,
):
    creds = get_and_validate_creds(user_id)

    # Parse user prompt
    num_posts, schedule, description, post_text = parse_user_prompt(user_prompt)
    print("Parsed user prompt:")
    print(num_posts, schedule, description, post_text)
    num_posts = min(num_posts, 10)  # Cap at 10 posts

    # Validate inputs
    if image_files and len(image_files) > num_posts:
        raise HTTPException(status_code=400, detail="Too many image files provided.")
    if urls and len(urls) > num_posts:
        raise HTTPException(status_code=400, detail="Too many URLs provided.")

    # Prepare posts
    posts = []
    print(f"url list: {urls}")
    for i in range(num_posts):
        text = generate_post_content(
            post_text, urls[i] if urls and i < len(urls) else None
        )

        print(f"Generated post text for post {i+1}: {text}")
        image_file = image_files[i] if image_files and i < len(image_files) else None
        url = urls[i] if urls and i < len(urls) else None

        # Save image file if provided
        image_path = None
        if image_file:
            image_path = os.path.join("images", f"{user_id}_{i}_{image_file.filename}")
            os.makedirs("images", exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

        post = {
            "user_id": user_id,
            "text": text,
            "image_path": image_path,
            "url": str(url) if url else None,
        }
        posts.append(post)

    # Handle scheduling or immediate posting
    if schedule:
        # Schedule posts
        timestamps = parse_schedule_to_utc(schedule, datetime.now(pytz.UTC))[:num_posts]
        scheduled_posts = load_scheduled_posts()
        posts_with_timestamp = [
            dict(post, timestamp=timestamps[i]) for i, post in enumerate(posts)
        ]
        scheduled_posts.extend(posts_with_timestamp)
        save_scheduled_posts(scheduled_posts)
        return JSONResponse(
            status_code=200,
            content={
                "message": f"{num_posts} posts scheduled successfully!",
                "scheduled_posts": posts_with_timestamp,
            },
        )
    else:
        # Post immediately
        results = []
        for post in posts:
            try:
                response = await universal_post(
                    user_id=user_id,
                    text=post["text"],
                    image_file=(
                        UploadFile(
                            filename=os.path.basename(post["image_path"]),
                            file=open(post["image_path"], "rb"),
                        )
                        if post["image_path"]
                        else None
                    ),
                    url=post["url"],
                )
                # Extract the JSON body from JSONResponse
                results.append(
                    response.body.decode("utf-8")
                    if isinstance(response, JSONResponse)
                    else response
                )
                # Clean up image file after posting
                if post["image_path"] and os.path.exists(post["image_path"]):
                    try:
                        os.remove(post["image_path"])
                    except OSError as e:
                        print(f"Error deleting image file {post['image_path']}: {e}")
            except Exception as e:
                results.append({"error": f"Failed to post: {str(e)}"})
        return JSONResponse(
            status_code=200,
            content={"message": f"{len(results)} posts processed", "results": results},
        )


@app.delete("/users/{user_id}/posts/{post_urn}", summary="Delete a LinkedIn Post")
async def delete_post(user_id: str, post_urn: str):
    creds = get_and_validate_creds(user_id)
    delete_url = f"{POSTS_API_URL}/{urllib.parse.quote(post_urn)}"
    print(f"Attempting to delete post at URL: {delete_url}")
    try:
        response = requests.delete(
            delete_url, headers=get_linkedin_headers(creds["access_token"])
        )
        response.raise_for_status()
        return JSONResponse(
            status_code=response.status_code,
            content={"message": f"Post {post_urn} deleted successfully."},
        )
    except requests.exceptions.RequestException as e:
        handle_linkedin_api_error(e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
