import json
import requests
import os
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SCHEDULED_POSTS_FILE = "scheduled_posts.json"
TOKEN_STORAGE_DIR = "user_tokens"
CHECK_INTERVAL_SECONDS = 1  # Check every second
TIME_WINDOW_SECONDS = 1  # Allow ±1 second for timestamp matching


def load_scheduled_posts() -> list[dict]:
    """Load scheduled posts from JSON file."""
    if not os.path.exists(SCHEDULED_POSTS_FILE):
        return []
    try:
        with open(SCHEDULED_POSTS_FILE, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading scheduled posts: {e}")
        return []


def save_scheduled_posts(posts: list[dict]):
    """Save updated scheduled posts to JSON file."""
    try:
        with open(SCHEDULED_POSTS_FILE, "w") as f:
            json.dump(posts, f, indent=4)
    except IOError as e:
        print(f"Error saving scheduled posts: {e}")


def load_user_credentials(user_id: str) -> dict | None:
    """Load user credentials from token storage."""
    file_path = os.path.join(TOKEN_STORAGE_DIR, f"{user_id}.json")
    if not os.path.exists(file_path):
        print(f"No credentials found for user {user_id}")
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading credentials for user {user_id}: {e}")
        return None


def execute_post(post: dict) -> bool:
    """Execute a single scheduled post and return True if successful."""
    try:
        user_id = post["user_id"]
        credentials = load_user_credentials(user_id)
        if not credentials or "access_token" not in credentials:
            print(f"Invalid credentials for user {user_id}. Skipping post.")
            return False

        # Prepare API call to /universal_post
        url = f"{API_BASE_URL}/users/{user_id}/universal_post"
        headers = {"Authorization": f"Bearer {credentials['access_token']}"}
        data = {"text": post["text"]}
        files = None

        if post["url"]:
            data["url"] = post["url"]

        if post["image_path"] and os.path.exists(post["image_path"]):
            files = {"image_file": open(post["image_path"], "rb")}
            # Ensure 'url' is not sent with image (per universal_post logic)
            data.pop("url", None)

        # Make API call
        response = requests.post(url, headers=headers, data=data, files=files)
        if files:
            files["image_file"].close()

        if response.status_code == 200:
            print(f"Successfully posted for user {user_id}: {response.json()}")
            # Delete local image file after posting
            if post["image_path"] and os.path.exists(post["image_path"]):
                try:
                    os.remove(post["image_path"])
                    print(f"Deleted image file: {post['image_path']}")
                except OSError as e:
                    print(f"Error deleting image file {post['image_path']}: {e}")
            return True
        else:
            print(
                f"Failed to post for user {user_id}: {response.status_code} {response.text}"
            )
            if response.status_code == 401:
                print(
                    f"Access token for user {user_id} may have expired. Post retained for retry."
                )
                return False
            return False

    except Exception as e:
        print(f"Error processing post for user {post['user_id']}: {e}")
        return False


def execute_scheduled_posts():
    """Continuously check and execute scheduled posts at their exact timestamps."""
    print(f"Starting scheduler at {datetime.now(pytz.UTC).isoformat()}")

    while True:
        posts = load_scheduled_posts()
        if not posts:
            print("No scheduled posts found. Waiting...")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        now = datetime.now(pytz.UTC)
        updated_posts = posts.copy()
        posts_to_remove = []

        for post in posts:
            try:
                post_time = datetime.fromisoformat(post["timestamp"])
                time_diff = (post_time - now).total_seconds()

                # Check if current time is within ±1 second of post_time
                if abs(time_diff) <= TIME_WINDOW_SECONDS:
                    if execute_post(post):
                        posts_to_remove.append(post)
                    # If token expired (401), retain post for retry
                elif time_diff < -TIME_WINDOW_SECONDS:
                    # Post is overdue (missed by more than 1 second), remove it
                    print(
                        f"Post for user {post['user_id']} at {post['timestamp']} is overdue. Removing."
                    )
                    posts_to_remove.append(post)

            except Exception as e:
                print(f"Error processing post for user {post['user_id']}: {e}")
                posts_to_remove.append(post)  # Remove to prevent infinite retries

        # Update scheduled posts
        for post in posts_to_remove:
            if post in updated_posts:
                updated_posts.remove(post)

        if updated_posts != posts:
            save_scheduled_posts(updated_posts)
            print(f"Updated scheduled posts. {len(updated_posts)} posts remaining.")

        # Sleep to avoid excessive CPU usage
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        execute_scheduled_posts()
    except KeyboardInterrupt:
        print("Scheduler stopped by user.")
# ```
# ### Overview of Changes
# This script has been updated to continuously check and execute scheduled posts at their exact timestamps, improving precision and resource usage. Here’s a summary of the key changes:
# 1. **Continuous Execution**:
#    - Replaced single-run logic with an infinite `while True` loop.
#    - Added `time.sleep(CHECK_INTERVAL_SECONDS)` to check every second, minimizing CPU usage.
# 2. **Precise Timestamp Matching**:
#    - Introduced `TIME_WINDOW_SECONDS = 1` to execute posts only when the current UTC time is within ±1 second of the post’s timestamp.
#    - Uses `time_diff = (post_time - now).total_seconds()` to compute the difference.
#    - Skips posts not yet due (`time_diff > TIME_WINDOW_SECONDS`).
#    - Removes overdue posts (`time_diff < -TIME_WINDOW_SECONDS`) to prevent posting stale content.
# 3. **Refactored Post Execution**:
#    - Moved post execution logic to a separate `execute_post` function for clarity.
#    - Returns `True` if the post succeeds, `False` otherwise (e.g., for 401 errors, posts are retained).
# 4. **Error Handling**:
#    - Catches `KeyboardInterrupt` to gracefully exit on Ctrl+C.
#    - Retains posts with 401 errors (token expiration) for potential retry after token refresh.
#    - Removes posts with other errors to avoid infinite retries.
# 5. **Logging**:
#    - Added startup message and periodic “No scheduled posts found” messages for clarity.
#    - Logs overdue post removals and errors.

# ### Setup Instructions
# 1. **Save the Script**:
#    - Replace `execute_scheduled_posts.py` with the updated version.
# 2. **Dependencies**:
#    - Ensure `requests`, `python-dotenv`, `pytz` are installed:
#      ```bash
#      pip install requests python-dotenv pytz
#      ```
# 3. **Configure `.env`**:
#    ```
#    API_BASE_URL=http://localhost:8000
#    ```
# 4. **Run the Script**:
#    - Start the scheduler:
#      ```bash
#      python execute_scheduled_posts.py
#      ```
#    - Runs continuously, checking timestamps every second.
#    - Stop with Ctrl+C.
# 5. **Optional: Run as a Service**:
#    - Use `systemd` or `pm2` to run in the background instead of cron, since it’s now continuous.
#    - Example `systemd` service file (`/etc/systemd/system/scheduler.service`):
#      ```
#      [Unit]
#      Description=LinkedIn Scheduled Posts Executor
#      After=network.target

#      [Service]
#      ExecStart=/usr/bin/python3 /path/to/execute_scheduled_posts.py
#      WorkingDirectory=/path/to/project
#      Restart=always
#      User=your_user
#      StandardOutput=append:/path/to/logs/scheduler.log
#      StandardError=append:/path/to/logs/scheduler.log

#      [Install]
#      WantedBy=multi-user.target
#      ```
#    - Enable and start:
#      ```bash
#      sudo systemctl enable scheduler.service
#      sudo systemctl start scheduler.service
#      ```

# ### Example Behavior
# **`scheduled_posts.json`**:
# ```json
# [
#     {
#         "user_id": "123456789",
#         "text": "Transform your business with Rapid Labs! https://rapidlabs.ai/ #AI",
#         "image_path": null,
#         "url": "https://rapidlabs.ai",
#         "timestamp": "2025-06-12T14:00:00+00:00"
#     }
# ]
# ```
# - **At 2:23 PM UTC (7:23 PM PKT), June 12, 2025**:
#   - Script checks every second.
#   - Skips the post (due at 2:00 PM UTC) as it’s overdue (`time_diff < -1`).
#   - Removes it from `scheduled_posts.json`.
#   - Logs: “Post for user 123456789 at 2025-06-12T14:00:00+00:00 is overdue. Removing.”
# - **At 2:00 PM UTC, June 13, 2025** (for a post scheduled at `2025-06-13T14:00:00+00:00`):
#   - When `now` is within ±1 second of 2:00 PM UTC, calls `/universal_post`.
#   - On success, removes the post and deletes any image.
#   - Logs: “Successfully posted for user 123456789: {...}”.

# ### Notes
# - **Precision**: The ±1-second window ensures posts are sent close to the exact timestamp. Adjust `TIME_WINDOW_SECONDS` if more leniency is needed.
# - **Overdue Posts**: Removed to avoid posting outdated content. If you want to retry overdue posts, remove the `time_diff < -TIME_WINDOW_SECONDS` condition.
# - **Token Expiration**: Posts with 401 errors are retained. Implement token refresh in `app.py` or re-authenticate manually.
# - **Resource Usage**: Checking every second is lightweight, but adjust `CHECK_INTERVAL_SECONDS` (e.g., to 5) for less frequent checks at the cost of precision.
# - **Timezone**: Uses UTC for consistency, as `scheduled_posts.json` timestamps are in UTC (from `app.py`’s `parse_schedule_to_utc`).

# If you need adjustments (e.g., handling overdue posts differently, adding token refresh, or testing with specific timestamps), let me know!
