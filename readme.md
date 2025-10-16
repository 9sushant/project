
How It Works
API Endpoint: A single /handle_task endpoint listens for POST requests.

Authentication: It validates a shared secret to ensure requests are legitimate.

Background Processing: It immediately acknowledges the request and starts the main workflow in a background task.

LLM Generation: It crafts a detailed prompt and asks an LLM (configurable for services like AIPipe, DeepInfra, etc.) to generate a single index.html file.

GitHub Deployment:

It creates a new public repository on GitHub.

It pushes the generated HTML, a README.md, and a LICENSE file.

It enables GitHub Pages for the repository.

It waits for the GitHub Pages site to become live.

Notification: It sends a POST request to a specified evaluation_url with the details of the deployment (repo URL, commit SHA, live site URL).

Setup & Installation
Follow these steps to get your server running.

1. Create a .env File
Create a file named .env in the project root. Copy the contents of .env.example into it and fill in your credentials.

IMPORTANT: Every line in the .env file must be a KEY="VALUE" pair or a comment starting with #. Do not leave any descriptive text uncommented.

2.(Recommended) Create a Virtual Environment
Using a virtual environment prevents conflicts with other projects by keeping dependencies separate.

Create the environment (only needs to be done once):

python3 -m venv venv

This will create a venv folder in your project directory.

Activate the environment (do this every time you start working):

source venv/bin/activate

Your terminal prompt will change to show (venv) at the beginning, indicating the environment is active.

3. Install Dependencies
With your virtual environment active, install the required Python packages.

pip install -r requirements.txt


3. Run the Server
Start the server using uvicorn. The --reload flag will automatically restart the server when you make changes to the code.

python3 -m uvicorn main:app --reload
#Crucially, check the entire repo scope AND the delete_repo scope.
The server will be running at http://127.0.0.1:8000.

4. Expose Your Server with ngrok
Since your server is running locally, you need a tool like ngrok to make it accessible from the public internet.

ngrok http 8000

Ngrok will give you a public "Forwarding" URL (e.g., https://random-string.ngrok.io). Your full API endpoint that you submit to the instructors will be this URL plus /handle_task.

 I Tested the Endpoint
 
