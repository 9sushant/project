import os
import requests
import base64
import subprocess
import time
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f">>> Loaded LLM_API_BASE_URL: {os.getenv('LLM_API_BASE_URL')}")
logging.info(f">>> Loaded SERVER_SECRET: {'SET' if os.getenv('SERVER_SECRET') else 'NOT SET'}")

# --- Environment Variables ---
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SERVER_SECRET = os.getenv("SERVER_SECRET")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://aipipe.org/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
REPO_NAME_TEMPLATE = os.getenv("REPO_NAME_TEMPLATE")  # e.g. "{task}-{nonce6}" or "myorg-{task}-{nonce}"

# --- Derived Constants ---
LLM_CHAT_COMPLETIONS_URL = f"{LLM_API_BASE_URL.rstrip('/')}/chat/completions"

# --- Pydantic Models ---
class Attachment(BaseModel):
    name: str
    url: str

class TaskData(BaseModel):
    email: str
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: List[str]
    evaluation_url: str
    attachments: Optional[List[Attachment]] = Field(default_factory=list)

# --- FastAPI App ---
app = FastAPI(title="LLM Code Deployment Service")

# --- Helpers ---
def validate_secret(secret: str) -> bool:
    if not SERVER_SECRET:
        logging.error("SERVER_SECRET not set in environment.")
        return False
    is_valid = secret == SERVER_SECRET
    logging.info(f"Secret validation result: {is_valid}")
    return is_valid

# repo name derivation helper (configurable)
def derive_repo_name(task: str, nonce: str, suffix_len: int = 6, template: Optional[str] = None) -> str:
    """
    Derive a repository name from task and nonce. Template may include placeholders: {task}, {nonce}, {nonce6}.
    If REPO_NAME_TEMPLATE env var is set, it overrides template param. Defaults to "{task}-{nonce6}".
    """
    tpl = template or REPO_NAME_TEMPLATE or "{task}-{nonce6}"
    nonce6 = nonce[:suffix_len]
    repo_name = tpl.replace("{task}", task).replace("{nonce}", nonce).replace("{nonce6}", nonce6)
    # sanitize: allow only alphanum, dash, underscore
    repo_name = "".join(c for c in repo_name if c.isalnum() or c in ['-', '_'])
    return repo_name


def _decode_attachment_content(att: Attachment) -> str:
    try:
        header, encoded = att.url.split(",", 1)
        data = base64.b64decode(encoded).decode("utf-8")
        return f"\n\n--- Attachment: {att.name} ---\n{data}\n--- End Attachment ---"
    except Exception as e:
        logging.warning(f"Failed to decode attachment {att.name}: {e}")
        return f"\n\n--- Attachment: {att.name} (failed to decode) ---"


def generate_code_with_llm(brief: str, checks: List[str], attachments: List[Attachment], existing_code: Optional[str] = None) -> str:
    """
    Generates (or revises) index.html via LLM. If existing_code is provided, instruct the LLM to update it
    according to the brief & checks rather than generating from scratch.
    """
    logging.info(f"Generating code using model: {LLM_MODEL}")
    attachment_content = ""
    if attachments:
        for att in attachments:
            attachment_content += _decode_attachment_content(att)

    if existing_code:
        prompt = f"""
You are an expert web developer whose task is to revise and improve an existing HTML file (index.html).
**Project Brief:** {brief}
**Checks:** {"; ".join(checks)}
**Attachments:** {attachment_content if attachment_content else "None"}

You are given the current index.html below, delimited by triple dashes.
Provide an updated, complete, self-contained HTML file (only the new index.html content — no markdown).
- If you change or fix things, prefer minimal and safe edits but ensure the checks and brief are satisfied.
- Keep CSS inside <style> in <head>, JS inside <script> at the end of <body>.
- If you need to add comments, keep them as HTML comments only.
- Return the full updated index.html file (not a diff, not explanatory text).

---BEGIN EXISTING INDEX.HTML---
{existing_code}
---END EXISTING INDEX.HTML---
"""
    else:
        prompt = f"""
You are an expert web developer. Create a complete, self-contained HTML file.
**Project Brief:** {brief}
**Checks:** {"; ".join(checks)}
**Attachments:** {attachment_content if attachment_content else "None"}
Instructions:
- Write everything in index.html
- Embed CSS in <style> in <head>
- Embed JS in <script> at end of <body>
- Do not include markdown, only pure HTML
"""

    headers = {
        "Authorization": f"Bearer {LLM_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096
    }

    try:
        logging.info(f"Sending LLM request to {LLM_CHAT_COMPLETIONS_URL}")
        res = requests.post(LLM_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=120)
        logging.info(f"LLM API status: {res.status_code}")
        logging.debug(f"LLM API response (first 500 chars): {res.text[:500]}")

        res.raise_for_status()
        result = res.json()
        generated_html = result["choices"][0]["message"]["content"]

        # Strip code fences if any
        if generated_html.strip().startswith("```"):
            generated_html = generated_html.strip()
            first_newline = generated_html.find("\n")
            if first_newline != -1:
                if generated_html.endswith("```"):
                    generated_html = generated_html[first_newline+1:-3].strip()
                else:
                    generated_html = generated_html[first_newline+1:].strip()

        logging.info("Successfully generated HTML from LLM.")
        return generated_html
    except Exception as e:
        logging.error(f"LLM API call failed: {e}", exc_info=True)
        raise


def create_github_repo(repo_name: str) -> bool:
    url = "https://api.github.com/user/repos"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    payload = {"name": repo_name, "private": False}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        logging.info(f"GitHub repo creation status: {res.status_code}")
        if res.status_code == 422:
            logging.warning(f"Repo {repo_name} may already exist.")
            return True
        res.raise_for_status()
        return True
    except requests.RequestException as e:
        logging.error(f"Repo creation failed: {e}")
        if 'res' in locals() and res is not None:
            logging.error(f"Response Body: {res.text}")
        return False


def run_git_command(cmd: List[str], cwd: str):
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        logging.debug(f"Git cmd ({' '.join(cmd)}) output: {result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {' '.join(cmd)}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        raise


def delete_github_repo(repo_name: str) -> bool:
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    try:
        res = requests.delete(url, headers=headers, timeout=20)
        if res.status_code in [204, 404]:
            logging.info(f"Repo {repo_name} deleted or did not exist (status {res.status_code}).")
            return True
        logging.error(f"Repo deletion failed: status {res.status_code} | body: {res.text}")
        return False
    except Exception as e:
        logging.error(f"Repo deletion error: {e}", exc_info=True)
        return False


def push_files_to_repo(repo_name: str, files: Dict[str, str], commit_msg: str, is_update: bool = False, dry_run: bool = False) -> str:
    """
    Push files to GitHub repo. If is_update=True, attempt to git clone the existing repo (preserving history),
    otherwise create a fresh local repo.
    If dry_run=True, skip any git/network operations and return a deterministic dry-run SHA.
    Returns the commit SHA of HEAD after push (or dry-run id).
    """
    local_path = f"/tmp/{repo_name}"

    # dry-run: skip all git and subprocess operations
    if dry_run:
        dry_sha = f"dryrun-{int(time.time())}"
        logging.info(f"DRY RUN enabled. Would push files to {repo_name}. Returning {dry_sha}")
        return dry_sha

    # Clean up if present
    if os.path.exists(local_path):
        subprocess.run(["rm", "-rf", local_path], check=True)

    repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"

    try:
        if is_update:
            logging.info(f"Attempting to clone existing repo {repo_name}")
            try:
                # clone into /tmp/{repo_name}
                run_git_command(["git", "clone", repo_url, local_path], cwd="/tmp")
            except Exception as e:
                logging.warning(f"Clone failed ({e}); falling back to init at {local_path}")
                os.makedirs(local_path, exist_ok=True)
                run_git_command(["git", "init"], cwd=local_path)
                run_git_command(["git", "branch", "-m", "main"], cwd=local_path)
                run_git_command(["git", "remote", "add", "origin", repo_url], cwd=local_path)
        else:
            logging.info(f"Initializing new repo at {local_path}")
            os.makedirs(local_path, exist_ok=True)
            run_git_command(["git", "init"], cwd=local_path)
            run_git_command(["git", "branch", "-m", "main"], cwd=local_path)
            run_git_command(["git", "remote", "add", "origin", repo_url], cwd=local_path)

        # Ensure git user config
        run_git_command(["git", "config", "user.name", GITHUB_USERNAME], cwd=local_path)
        run_git_command(["git", "config", "user.email", "bot@example.com"], cwd=local_path)

        # Write files
        for filename, content in files.items():
            file_path = os.path.join(local_path, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Add & commit
        run_git_command(["git", "add", "."], cwd=local_path)
        try:
            run_git_command(["git", "commit", "-m", commit_msg], cwd=local_path)
        except Exception as e:
            logging.info(f"No changes to commit or commit failed: {e}")

        # Ensure remote is set correctly (replace origin if needed)
        try:
            run_git_command(["git", "remote", "set-url", "origin", repo_url], cwd=local_path)
        except Exception:
            run_git_command(["git", "remote", "add", "origin", repo_url], cwd=local_path)

        # Make sure branch exists and push
        try:
            run_git_command(["git", "checkout", "-B", "main"], cwd=local_path)
        except Exception:
            logging.warning("Could not ensure local main branch, continuing.")

        run_git_command(["git", "push", "-f", "-u", "origin", "main"], cwd=local_path)

        commit_sha = run_git_command(["git", "rev-parse", "HEAD"], cwd=local_path)
        logging.info(f"Pushed commit {commit_sha} to {repo_name}")

        # Clean up
        subprocess.run(["rm", "-rf", local_path], check=True)
        return commit_sha
    except Exception as e:
        logging.error(f"Push to repo failed: {e}", exc_info=True)
        if os.path.exists(local_path):
            try:
                subprocess.run(["rm", "-rf", local_path], check=True)
            except Exception:
                pass
        raise


def enable_github_pages(repo_name: str) -> bool:
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.switcheroo-preview+json"}
    try:
        res = requests.post(url, headers=headers, json={"source": {"branch": "main"}}, timeout=20)
        logging.info(f"GitHub Pages status: {res.status_code}")
        if res.status_code == 409:  # Conflict = already enabled or configuration conflict
            logging.warning("GitHub Pages is already enabled or in conflict.")
            return True
        res.raise_for_status()
        return True
    except requests.RequestException as e:
        logging.error(f"GitHub Pages enabling failed: {e}")
        if 'res' in locals() and res is not None:
            logging.error(f"Response Body: {res.text}")
        return False


def wait_for_pages_deployment(pages_url: str, timeout: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(pages_url, timeout=10)
            if r.status_code == 200:
                logging.info(f"Pages URL live: {pages_url}")
                return True
        except Exception:
            pass
        time.sleep(10)
    logging.error("Pages deployment timeout reached.")
    return False


def notify_evaluation_api(url: str, payload: dict):
    delay = 1
    for _ in range(5):
        try:
            res = requests.post(url, json=payload, timeout=20)
            if res.status_code == 200:
                logging.info("Evaluation API notified successfully.")
                return
            logging.warning(f"Evaluation API returned status {res.status_code} - {res.text}")
        except Exception as e:
            logging.warning(f"Notify evaluation API failed: {e}")
        time.sleep(delay)
        delay *= 2
    logging.error("Failed to notify evaluation API after retries.")


def get_file_from_repo(repo_name: str, path: str = "index.html") -> Optional[str]:
    """
    Fetch a file's raw content from the GitHub repository.
    """
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{path}"
    # This header asks for the raw file, not a JSON response
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}
    try:
        res = requests.get(url, headers=headers, timeout=20)
        logging.info(f"Get file from repo status: {res.status_code} for {repo_name}/{path}")
        
        if res.status_code == 200:
            # The response is raw text, so we use .text directly. No .json() needed.
            return res.text
        
        elif res.status_code == 404:
            logging.warning(f"File not found in repo: {repo_name}/{path}")
            return None
        else:
            logging.error(f"Failed to fetch file: {res.status_code} | {res.text}")
            return None
            
    except Exception as e:
        # The original traceback shows the error is caught here
        logging.error(f"Error fetching file from repo: {e}", exc_info=True)
        return None

# --- Background Tasks ---
def round1_task(data: TaskData):
    repo_name = derive_repo_name(data.task, data.nonce)
    logging.info(f"--- Starting Round 1 task: {repo_name} ---")
    try:
        html_code = generate_code_with_llm(data.brief, data.checks, data.attachments)
        readme = f"# {repo_name}\n\n**Brief:**\n{data.brief}"
        license_content = "MIT License — see LICENSE file for details."

        logging.info(f"Ensuring repo {repo_name} does not exist before creation.")
        delete_success = delete_github_repo(repo_name)
        if not delete_success:
            logging.warning("Delete attempt returned False; continuing to try create.")

        if not create_github_repo(repo_name):
            raise Exception("Repo creation failed definitively.")

        commit_sha = push_files_to_repo(repo_name, {
            "index.html": html_code,
            "README.md": readme,
            "LICENSE": license_content
        }, "feat: Initial application build")

        if not enable_github_pages(repo_name):
            raise Exception("Pages enabling failed.")

        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        if not wait_for_pages_deployment(pages_url):
            raise Exception("Pages deployment timeout.")

        notify_evaluation_api(data.evaluation_url, {
            "email": data.email,
            "task": data.task,
            "round": 1,
            "nonce": data.nonce,
            "repo_url": f"https://github.com/{GITHUB_USERNAME}/{repo_name}",
            "commit_sha": commit_sha,
            "pages_url": pages_url
        })

        logging.info(f"--- Finished Round 1 Task: {repo_name} ---")
    except Exception as e:
        logging.error(f"Round 1 failed: {e}", exc_info=True)


def round2_task(data: TaskData):
    try:
        repo_name = find_existing_repo(data.task)
        logging.info(f"--- Starting Round 2 revise task: {repo_name} ---")

        existing_html = get_file_from_repo(repo_name, "index.html")
        if existing_html is None:
            logging.warning("Existing index.html not found; Round 2 will act like a fresh generation.")

        revised_html = generate_code_with_llm(
            data.brief,
            data.checks,
            data.attachments,
            existing_code=existing_html
        )

        commit_sha = push_files_to_repo(
            repo_name,
            {
                "index.html": revised_html,
                "README.md": f"# {repo_name}\n\n**Brief:**\n{data.brief}\n\n**Round 2 update**",
            },
            "chore: Revise application (round 2)",
            is_update=True
        )

        if not enable_github_pages(repo_name):
            logging.warning("Failed to enable GitHub Pages after update; continuing to notify evaluation.")

        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        if not wait_for_pages_deployment(pages_url, timeout=300):
            logging.warning("Pages did not become live within timeout after update.")

        notify_payload = {
            "email": data.email,
            "task": data.task,
            "round": 2,
            "nonce": data.nonce,
            "repo_url": f"https://github.com/{GITHUB_USERNAME}/{repo_name}",
            "commit_sha": commit_sha,
            "pages_url": pages_url
        }
        notify_evaluation_api(data.evaluation_url, notify_payload)

        logging.info(f"--- Finished Round 2 Task: {repo_name} ---")

    except Exception as e:
        logging.error(f"Round 2 failed: {e}", exc_info=True)


def find_existing_repo(task_name: str) -> str:
    """
    Finds a repository by its prefix, handling paginated results from the GitHub API.
    """
    url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos?per_page=100" # Increased page size
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    while url:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            repos = response.json()

            for repo in repos:
                if repo["name"].startswith(task_name):
                    logging.info(f"Found existing repo: {repo['name']}")
                    return repo["name"]

            # Check for the 'next' page link in the response headers
            if 'Link' in response.headers:
                links = requests.utils.parse_header_links(response.headers['Link'])
                next_link = next((link for link in links if link.get('rel') == 'next'), None)
                url = next_link['url'] if next_link else None
            else:
                url = None # No more pages

        except requests.RequestException as e:
            logging.error(f"Error fetching repos from GitHub: {e}")
            raise ValueError(f"Could not search for repo due to API error: {e}")

    # This is reached only after checking all pages and finding no match
    raise ValueError(f"No existing repo found for task '{task_name}'")
        

# --- Routes ---
@app.post("/handle_task")
async def handle_task(data: TaskData, background_tasks: BackgroundTasks):
    logging.info(f"Received task: {data.task} | Round: {data.round}")
    if not validate_secret(data.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")

    if data.round == 1:
        background_tasks.add_task(round1_task, data)
        return {"message": "Round 1 task started."}
    elif data.round == 2:
        background_tasks.add_task(round2_task, data)
        return {"message": "Round 2 task started."}
    raise HTTPException(status_code=400, detail="Invalid round number")

@app.get("/")
def root():
    return {"status": "Service is running."}






# import os
# import requests
# import base64
# import subprocess
# import time
# import logging
# from fastapi import FastAPI, BackgroundTasks, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Optional
# from dotenv import load_dotenv

# # --- Load environment variables ---
# load_dotenv()
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logging.info(f">>> Loaded LLM_API_BASE_URL: {os.getenv('LLM_API_BASE_URL')}")
# logging.info(f">>> Loaded SERVER_SECRET: {'SET' if os.getenv('SERVER_SECRET') else 'NOT SET'}")

# # --- Environment Variables ---
# GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# SERVER_SECRET = os.getenv("SERVER_SECRET")
# LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")
# LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://aipipe.org/openai/v1")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")

# # --- Derived Constants ---
# LLM_CHAT_COMPLETIONS_URL = f"{LLM_API_BASE_URL.rstrip('/')}/chat/completions"

# # --- Pydantic Models ---
# class Attachment(BaseModel):
#     name: str
#     url: str

# class TaskData(BaseModel):
#     email: str
#     secret: str
#     task: str
#     round: int
#     nonce: str
#     brief: str
#     checks: List[str]
#     evaluation_url: str
#     attachments: Optional[List[Attachment]] = Field(default_factory=list)

# # --- FastAPI App ---
# app = FastAPI(title="LLM Code Deployment Service")

# # --- Helpers ---
# def validate_secret(secret: str) -> bool:
#     if not SERVER_SECRET:
#         logging.error("SERVER_SECRET not set in environment.")
#         return False
#     is_valid = secret == SERVER_SECRET
#     logging.info(f"Secret validation result: {is_valid}")
#     return is_valid

# def generate_code_with_llm(brief: str, checks: List[str], attachments: List[Attachment]) -> str:
#     logging.info(f"Generating code using model: {LLM_MODEL}")
#     attachment_content = ""
#     if attachments:
#         for att in attachments:
#             try:
#                 header, encoded = att.url.split(",", 1)
#                 data = base64.b64decode(encoded).decode("utf-8")
#                 attachment_content += f"\n\n--- Attachment: {att.name} ---\n{data}\n--- End Attachment ---"
#             except Exception as e:
#                 logging.warning(f"Failed to decode attachment {att.name}: {e}")

#     prompt = f"""
# You are an expert web developer. Create a complete, self-contained HTML file.
# **Project Brief:** {brief}
# **Checks:** {"; ".join(checks)}
# **Attachments:** {attachment_content if attachment_content else "None"}
# Instructions:
# - Write everything in index.html
# - Embed CSS in <style> in <head>
# - Embed JS in <script> at end of <body>
# - Do not include markdown, only pure HTML
# """

#     headers = {
#         "Authorization": f"Bearer {LLM_API_TOKEN}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": LLM_MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": 4096
#     }

#     try:
#         logging.info(f"Sending LLM request to {LLM_CHAT_COMPLETIONS_URL}")
#         res = requests.post(LLM_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=120)
#         logging.info(f"LLM API status: {res.status_code}")
#         logging.info(f"LLM API response (first 500 chars): {res.text[:500]}")

#         res.raise_for_status()
#         result = res.json()
#         generated_html = result["choices"][0]["message"]["content"]

#         if generated_html.strip().startswith("```html"):
#             generated_html = generated_html.strip()[7:-3].strip()

#         logging.info("Successfully generated HTML from LLM.")
#         return generated_html
#     except Exception as e:
#         logging.error(f"LLM API call failed: {e}", exc_info=True)
#         raise

# def create_github_repo(repo_name: str) -> bool:
#     url = "https://api.github.com/user/repos"
#     headers = {"Authorization": f"token {GITHUB_TOKEN}"}
#     payload = {"name": repo_name, "private": False}
#     try:
#         res = requests.post(url, headers=headers, json=payload, timeout=20)
#         logging.info(f"GitHub repo creation status: {res.status_code}")
#         if res.status_code == 422:
#             logging.warning(f"Repo {repo_name} may already exist.")
#             return True
#         res.raise_for_status()
#         return True
#     except requests.RequestException as e:
#         logging.error(f"Repo creation failed: {e}")
#         if 'res' in locals() and res is not None:
#             logging.error(f"Response Body: {res.text}")
#         return False

# def run_git_command(cmd: List[str], cwd: str):
#     try:
#         result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
#         logging.debug(f"Git cmd output: {result.stdout.strip()}")
#         return result.stdout.strip()
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Git command failed: {' '.join(cmd)}\n{e.stderr}")
#         raise

# def delete_github_repo(repo_name: str) -> bool:
#     # proper repo delete endpoint
#     url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}"
#     headers = {"Authorization": f"token {GITHUB_TOKEN}"}
#     try:
#         res = requests.delete(url, headers=headers, timeout=20)
#         # 204 = deleted, 404 = not found (already absent)
#         if res.status_code in [204, 404]:
#             logging.info(f"Repo {repo_name} deleted or did not exist (status {res.status_code}).")
#             return True
#         logging.error(f"Repo deletion failed: status {res.status_code} | body: {res.text}")
#         return False
#     except Exception as e:
#         logging.error(f"Repo deletion error: {e}", exc_info=True)
#         return False

# def push_files_to_repo(repo_name: str, files: dict, commit_msg: str) -> str:
#     local_path = f"/tmp/{repo_name}"
#     if os.path.exists(local_path):
#         subprocess.run(["rm", "-rf", local_path], check=True)
#     os.makedirs(local_path, exist_ok=True)

#     run_git_command(["git", "init"], cwd=local_path)
#     run_git_command(["git", "branch", "-m", "main"], cwd=local_path)
#     run_git_command(["git", "config", "user.name", GITHUB_USERNAME], cwd=local_path)
#     run_git_command(["git", "config", "user.email", "bot@example.com"], cwd=local_path)

#     for filename, content in files.items():
#         with open(os.path.join(local_path, filename), "w") as f:
#             f.write(content)

#     run_git_command(["git", "add", "."], cwd=local_path)
#     run_git_command(["git", "commit", "-m", commit_msg], cwd=local_path)

#     # correct remote URL (no markdown)
#     repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"

#     # add or update origin
#     try:
#         run_git_command(["git", "remote", "add", "origin", repo_url], cwd=local_path)
#     except Exception:
#         run_git_command(["git", "remote", "set-url", "origin", repo_url], cwd=local_path)

#     # try fetching remote to detect existing history, reset if present (safe)
#     try:
#         run_git_command(["git", "fetch", "origin"], cwd=local_path)
#         # if origin/main exists, reset local to that so push will succeed as fast-forward or force
#         run_git_command(["git", "reset", "--hard", "origin/main"], cwd=local_path)
#     except Exception as e:
#         logging.warning(f"Remote fetch/reset skipped or no remote main: {e}")

#     # force push to ensure remote matches local (CI-friendly)
#     run_git_command(["git", "push", "-f", "-u", "origin", "main"], cwd=local_path)
#     commit_sha = run_git_command(["git", "rev-parse", "HEAD"], cwd=local_path)

#     subprocess.run(["rm", "-rf", local_path], check=True)
#     return commit_sha

# def enable_github_pages(repo_name: str) -> bool:
#     url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages"
#     headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.switcheroo-preview+json"}
#     try:
#         res = requests.post(url, headers=headers, json={"source": {"branch": "main"}}, timeout=20)
#         logging.info(f"GitHub Pages status: {res.status_code}")
#         if res.status_code == 409:  # Conflict = already enabled or configuration conflict
#             logging.warning("GitHub Pages is already enabled or in conflict.")
#             return True
#         res.raise_for_status()
#         return True
#     except requests.RequestException as e:
#         logging.error(f"GitHub Pages enabling failed: {e}")
#         if 'res' in locals() and res is not None:
#             logging.error(f"Response Body: {res.text}")
#         return False

# def wait_for_pages_deployment(pages_url: str, timeout: int = 300) -> bool:
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             r = requests.get(pages_url, timeout=10)
#             if r.status_code == 200:
#                 logging.info(f"Pages URL live: {pages_url}")
#                 return True
#         except Exception:
#             pass
#         time.sleep(10)
#     logging.error("Pages deployment timeout reached.")
#     return False

# def notify_evaluation_api(url: str, payload: dict):
#     delay = 1
#     for _ in range(5):
#         try:
#             res = requests.post(url, json=payload, timeout=20)
#             if res.status_code == 200:
#                 logging.info("Evaluation API notified successfully.")
#                 return
#             logging.warning(f"Evaluation API returned status {res.status_code} - {res.text}")
#         except Exception as e:
#             logging.warning(f"Notify evaluation API failed: {e}")
#         time.sleep(delay)
#         delay *= 2
#     logging.error("Failed to notify evaluation API after retries.")

# # --- Background Tasks ---
# def round1_task(data: TaskData):
#     # include a small unique suffix to avoid collisions, optional
#     repo_name = f"{data.task}-{data.nonce[:6]}"
#     logging.info(f"--- Starting Round 1 task: {repo_name} ---")
#     try:
#         html_code = generate_code_with_llm(data.brief, data.checks, data.attachments)
#         readme = f"# {repo_name}\n\n**Brief:**\n{data.brief}"
#         license_content = "MIT License — see LICENSE file for details."

#         # Ensure repo is absent (delete if exists) to get a clean push
#         logging.info(f"Ensuring repo {repo_name} does not exist before creation.")
#         delete_success = delete_github_repo(repo_name)
#         if not delete_success:
#             logging.warning("Delete attempt returned False; continuing to try create.")

#         # create repo
#         if not create_github_repo(repo_name):
#             raise Exception("Repo creation failed definitively.")

#         # push files
#         commit_sha = push_files_to_repo(repo_name, {
#             "index.html": html_code,
#             "README.md": readme,
#             "LICENSE": license_content
#         }, "feat: Initial application build")

#         # enable pages and wait for deployment
#         if not enable_github_pages(repo_name):
#             raise Exception("Pages enabling failed.")

#         pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
#         if not wait_for_pages_deployment(pages_url):
#             raise Exception("Pages deployment timeout.")

#         # notify evaluation
#         notify_evaluation_api(data.evaluation_url, {
#             "email": data.email,
#             "task": data.task,
#             "round": 1,
#             "nonce": data.nonce,
#             "repo_url": f"https://github.com/{GITHUB_USERNAME}/{repo_name}",
#             "commit_sha": commit_sha,
#             "pages_url": pages_url
#         })

#         logging.info(f"--- Finished Round 1 Task: {repo_name} ---")
#     except Exception as e:
#         logging.error(f"Round 1 failed: {e}", exc_info=True)

# def round2_task(data: TaskData):
#     logging.info("Round 2 is not implemented yet.")

# # --- Routes ---
# @app.post("/handle_task")
# async def handle_task(data: TaskData, background_tasks: BackgroundTasks):
#     logging.info(f"Received task: {data.task} | Round: {data.round}")
#     if not validate_secret(data.secret):
#         raise HTTPException(status_code=403, detail="Invalid secret")

#     if data.round == 1:
#         background_tasks.add_task(round1_task, data)
#         return {"message": "Round 1 task started."}
#     elif data.round == 2:
#         background_tasks.add_task(round2_task, data)
#         return {"message": "Round 2 task started."}
#     raise HTTPException(status_code=400, detail="Invalid round number")

# @app.get("/")
# def root():
#     return {"status": "Service is running."}
