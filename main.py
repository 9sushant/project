import os
import requests
import base64
import subprocess
import time
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file for security
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Variables ---
# GitHub Credentials
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Server Secret (shared with instructors)
SERVER_SECRET = os.getenv("SERVER_SECRET")

# LLM Provider Configuration (configurable for AIPipe, DeepInfra, etc.)
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.deepinfra.com/v1") # Default to DeepInfra
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3-8b-instruct") # Default to a good model on DeepInfra

# --- Constants ---
LLM_CHAT_COMPLETIONS_URL = f"{LLM_API_BASE_URL}/openai/chat/completions"


# --- Pydantic Models for Data Validation ---
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


# --- FastAPI Application Instance ---
app = FastAPI(
    title="LLM Code Deployment Service",
    description="An automated service to build, deploy, and update web applications using LLMs."
)


# --- Core Logic Functions ---

def validate_secret(secret: str) -> bool:
    """Validates the incoming secret against the one stored in environment variables."""
    if not SERVER_SECRET:
        logging.error("SERVER_SECRET is not set in the environment.")
        return False
    return secret == SERVER_SECRET

def generate_code_with_llm(brief: str, checks: List[str], attachments: List[Attachment]) -> str:
    """Generates a single HTML file using an LLM based on the project brief."""
    logging.info(f"Generating code with LLM model: {LLM_MODEL}")

    # 1. Decode attachments and prepare them for the prompt
    attachment_content = ""
    if attachments:
        for att in attachments:
            try:
                header, encoded = att.url.split(",", 1)
                data = base64.b64decode(encoded).decode('utf-8')
                attachment_content += f"\n\n--- Attachment: {att.name} ---\n{data}\n--- End Attachment ---"
            except Exception as e:
                logging.warning(f"Could not decode attachment {att.name}: {e}")

    # 2. Craft a detailed prompt for the LLM
    prompt = f"""
    You are an expert web developer. Your task is to create a complete, self-contained, single HTML file.
    This file must not have any external dependencies unless explicitly requested in the brief.
    All CSS and JavaScript must be embedded directly within the HTML file.

    **Project Brief:**
    {brief}

    **The application will be evaluated against these checks:**
    - {"\n- ".join(checks)}

    **Attachments Content:**
    {attachment_content if attachment_content else "No attachments provided."}

    **Instructions:**
    - Write all code in a single index.html file.
    - Embed CSS in a `<style>` tag in the `<head>`.
    - Embed JavaScript in a `<script>` tag at the end of the `<body>`.
    - Ensure the code directly addresses all points in the brief and is likely to pass the checks.
    - Provide only the full HTML code as your response, without any explanations or markdown formatting.
    """

    # 3. Call the configured LLM API (works with any OpenAI-compatible endpoint)
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
        response = requests.post(LLM_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        generated_html = result['choices'][0]['message']['content']
        # Clean up potential markdown code blocks from the LLM's response
        if generated_html.strip().startswith("```html"):
            generated_html = generated_html.strip()[7:-3].strip()
        logging.info("Successfully generated code from LLM.")
        return generated_html
    except requests.RequestException as e:
        logging.error(f"Error calling LLM API: {e}")
        raise

def create_github_repo(repo_name: str) -> bool:
    """Creates a public GitHub repository."""
    logging.info(f"Creating GitHub repo: {repo_name}")
    url = "[https://api.github.com/user/repos](https://api.github.com/user/repos)"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    payload = {
        "name": repo_name,
        "private": False,
        "auto_init": False, # We will init it locally
        "description": f"Repository for task: {repo_name}"
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        # 422 (Unprocessable Entity) often means the repo already exists on GitHub's end.
        if response.status_code == 422:
            logging.warning(f"Repo {repo_name} might already exist.")
            return True
        response.raise_for_status()
        logging.info(f"Successfully created GitHub repo: {repo_name}")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to create GitHub repo: {response.text}")
        return False

def run_git_command(command: list, cwd: str):
    """Helper to run a git command in a specific directory."""
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
        logging.info(f"Git command `{' '.join(command)}` successful.")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command `{' '.join(command)}` failed.")
        logging.error(f"Stderr: {e.stderr}")
        logging.error(f"Stdout: {e.stdout}")
        raise

def push_files_to_repo(repo_name: str, files: dict, commit_message: str):
    """Initializes a repo locally, adds files, commits, and pushes to GitHub."""
    logging.info(f"Initializing and pushing files to {repo_name}")
    repo_url_with_token = f"https://{GITHUB_TOKEN}@[github.com/](https://github.com/){GITHUB_USERNAME}/{repo_name}.git"
    local_path = f"/tmp/{repo_name}"

    # Clean up previous directory if it exists
    if os.path.exists(local_path):
        subprocess.run(["rm", "-rf", local_path], check=True)
    
    os.makedirs(local_path, exist_ok=True)

    # Git operations
    run_git_command(["git", "init"], cwd=local_path)
    run_git_command(["git", "branch", "-m", "main"], cwd=local_path)
    run_git_command(["git", "config", "user.name", GITHUB_USERNAME], cwd=local_path)
    run_git_command(["git", "config", "user.email", "bot@example.com"], cwd=local_path)

    # Write files
    for filename, content in files.items():
        with open(os.path.join(local_path, filename), "w") as f:
            f.write(content)
    
    run_git_command(["git", "add", "."], cwd=local_path)
    run_git_command(["git", "commit", "-m", commit_message], cwd=local_path)
    run_git_command(["git", "remote", "add", "origin", repo_url_with_token], cwd=local_path)
    run_git_command(["git", "push", "-u", "origin", "main"], cwd=local_path)
    
    # Get commit SHA
    commit_sha = run_git_command(["git", "rev-parse", "HEAD"], cwd=local_path).stdout.strip()
    
    # Clean up local clone
    subprocess.run(["rm", "-rf", local_path], check=True)
    
    logging.info(f"Successfully pushed files to {repo_name}. Commit SHA: {commit_sha}")
    return commit_sha

def enable_github_pages(repo_name: str) -> bool:
    """Enables GitHub Pages for the repository."""
    logging.info(f"Enabling GitHub Pages for {repo_name}")
    url = f"[https://api.github.com/repos/](https://api.github.com/repos/){GITHUB_USERNAME}/{repo_name}/pages"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.switcheroo-preview+json"
    }
    payload = {"source": {"branch": "main", "path": "/"}}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logging.info("GitHub Pages enabled successfully.")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to enable GitHub Pages: {response.text}")
        return False

def wait_for_pages_deployment(pages_url: str, timeout: int = 300) -> bool:
    """Waits for the GitHub Pages URL to become live."""
    logging.info(f"Waiting for Pages URL to be live: {pages_url}")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(pages_url, timeout=10)
            if response.status_code == 200:
                logging.info("GitHub Pages is live!")
                return True
        except requests.RequestException:
            pass # Ignore connection errors while waiting
        time.sleep(15)
    logging.error("Timeout reached while waiting for GitHub Pages.")
    return False

def notify_evaluation_api(url: str, payload: dict):
    """Notifies the evaluation API with the results, with exponential backoff."""
    logging.info(f"Notifying evaluation API at {url}")
    delay = 1
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                logging.info("Successfully notified evaluation API.")
                return
            logging.warning(f"Evaluation API returned status {response.status_code}. Retrying...")
        except requests.RequestException as e:
            logging.warning(f"Could not connect to evaluation API: {e}. Retrying...")
        
        time.sleep(delay)
        delay *= 2
    logging.error("Failed to notify evaluation API after several retries.")


# --- Background Task Definitions ---

def round1_task(data: TaskData):
    """The complete workflow for a Round 1 task."""
    repo_name = f"{data.task}-{data.nonce[:6]}"
    logging.info(f"--- Starting Round 1 Task: {repo_name} ---")

    try:
        # 1. Generate code using LLM
        html_code = generate_code_with_llm(data.brief, data.checks, data.attachments)
        readme_content = f"# {repo_name}\n\n**Brief:**\n{data.brief}"
        # In a real scenario, you'd fetch the full MIT license text.
        license_content = """
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        # 2. Create GitHub repo
        if not create_github_repo(repo_name):
            raise Exception("Failed to create GitHub repository.")

        # 3. Push generated files
        files_to_push = {
            "index.html": html_code,
            "README.md": readme_content,
            "LICENSE": license_content
        }
        commit_sha = push_files_to_repo(repo_name, files_to_push, "feat: Initial application build")

        # 4. Enable GitHub Pages
        if not enable_github_pages(repo_name):
            raise Exception("Failed to enable GitHub Pages.")

        # 5. Wait for deployment
        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        if not wait_for_pages_deployment(pages_url):
            raise Exception("GitHub Pages did not deploy in time.")

        # 6. Notify evaluation server
        eval_payload = {
            "email": data.email,
            "task": data.task,
            "round": 1,
            "nonce": data.nonce,
            "repo_url": f"[https://github.com/](https://github.com/){GITHUB_USERNAME}/{repo_name}",
            "commit_sha": commit_sha,
            "pages_url": pages_url
        }
        notify_evaluation_api(data.evaluation_url, eval_payload)

    except Exception as e:
        logging.error(f"Round 1 task for {repo_name} failed: {e}", exc_info=True)
    
    logging.info(f"--- Finished Round 1 Task: {repo_name} ---")

def round2_task(data: TaskData):
    """The complete workflow for a Round 2 task."""
    logging.info(f"--- Starting Round 2 Task: {data.task} ---")
    logging.warning("Round 2 task is not fully implemented.")
    # TODO:
    # 1. Fetch existing code from the GitHub repo.
    # 2. Pass existing code + new brief to the LLM.
    # 3. Push updated files to the repo.
    # 4. Notify evaluation API with round: 2.
    logging.info(f"--- Finished Round 2 Task: {data.task} ---")


# --- API Endpoint ---

@app.post("/handle_task")
async def handle_task(data: TaskData, background_tasks: BackgroundTasks):
    """
    This is the main endpoint that receives task requests.
    It validates the secret, acknowledges the request immediately,
    and starts the processing in the background.
    """
    if not validate_secret(data.secret):
        raise HTTPException(status_code=403, detail="Invalid secret")

    if data.round == 1:
        background_tasks.add_task(round1_task, data)
        return {"message": "Round 1 task received and started."}
    elif data.round == 2:
        background_tasks.add_task(round2_task, data)
        return {"message": "Round 2 task received and started."}
    else:
        raise HTTPException(status_code=400, detail="Invalid round number")

@app.get("/")
def read_root():
    return {"status": "Service is running."}

# To run this app: `uvicorn main:app --reload`