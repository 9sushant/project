# ----------------------------
# Basic unit-tests / smoke-tests
# Run with: pytest main.py -q
# These tests mock network and subprocess interactions so they won't hit real GitHub.
# ----------------------------

import pytest
from unittest.mock import patch, MagicMock

# Helper to produce a mock response object
class MockResponse:
    def __init__(self, status_code=200, json_data=None, text_data=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.HTTPError(f"Status {self.status_code}")


def test_get_file_from_repo_success(monkeypatch):
    # prepare base64 content
    html = '<html><body>ok</body></html>'
    encoded = base64.b64encode(html.encode()).decode()
    mock_json = {"content": encoded}

    def mock_get(url, headers, timeout):
        return MockResponse(status_code=200, json_data=mock_json)

    monkeypatch.setattr(requests, "get", mock_get)

    content = get_file_from_repo("some-repo", "index.html")
    assert content is not None
    assert "ok" in content


def test_get_file_from_repo_not_found(monkeypatch):
    def mock_get(url, headers, timeout):
        return MockResponse(status_code=404, json_data={})

    monkeypatch.setattr(requests, "get", mock_get)
    content = get_file_from_repo("no-repo", "index.html")
    assert content is None


def test_push_files_to_repo_dry_run():
    # dry_run should not attempt git operations
    sha = push_files_to_repo("repo-x", {"index.html": "<html/>"}, "msg", dry_run=True)
    assert sha.startswith("dryrun-")


@patch("subprocess.run")
def test_push_files_to_repo_init_and_push(mock_subprocess_run, monkeypatch):
    # Simulate successful subprocess.run and run_git_command
    # run_git_command is used heavily; patch it to avoid real git calls
    def fake_run_git_command(cmd, cwd):
        # Return predictable outputs for rev-parse
        if cmd[:2] == ["git", "rev-parse"]:
            return "deadbeef"
        return "ok"

    monkeypatch.setattr('builtins.__xonsh__', None, raising=False)
    monkeypatch.setattr(__name__ + '.run_git_command', fake_run_git_command)

    # Make subprocess.run a no-op (used for rm -rf calls)
    mock_subprocess_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    # Create a temporary directory to be used by the function
    # Run function (it should not raise)
    sha = push_files_to_repo("repo-x", {"index.html": "<html/>"}, "msg", is_update=False)
    assert sha == "deadbeef"


def test_derive_repo_name_defaults():
    name = derive_repo_name("taskname", "noncevalue12345")
    assert name.startswith("taskname-")


def test_derive_repo_name_template():
    name = derive_repo_name("t", "n123456", template="prefix-{task}-{nonce}")
    assert name.startswith("prefix-t-n123456")


if __name__ == "__main__":
    print("This module provides the FastAPI app and unit-tests. Run 'pytest main.py -q' to execute tests.")
