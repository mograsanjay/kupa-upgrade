#!/usr/bin/env python3

from requests.utils import prepend_scheme_if_needed
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
import uvicorn
import yaml
import os
from git import Repo
import tempfile
import shutil
from dotenv import load_dotenv
from kubernetes import client, config
from concurrent.futures import ThreadPoolExecutor
import openai
import re
from datetime import datetime
from yaml.loader import SafeLoader
from yaml.constructor import Constructor
from yaml.nodes import ScalarNode

app = FastAPI(title="MCP Server", description="Kubernetes Model Context Protocol Server")

class GitHubConfig(BaseModel):
    manifest_path: str = 'manifests'
    scan_path: Optional[str] = None  # If None, start from repo base; otherwise, start from specified path
    repo_url: str
    github_token: str
    github_branch: str = 'main'
    openai_api_key: str

class ManifestUpdate(BaseModel):
    manifest: Dict[str, Any]
    file_path: str

class ManifestAnalysis(BaseModel):
    manifest: Dict[str, Any]
    #issues: List[str]
    #suggestions: List[str]

class MCPServer:
    def __init__(self, openai_api_key: str):
        # Initialize AI client (OpenAI or Ollama)
        self.use_ollama = not openai_api_key
        if not self.use_ollama:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("Successfully initialized OpenAI client")
        else:
            import ollama
            self.ollama_client = ollama
            logger.info("Using local Ollama model for analysis")
        
        # Initialize Helm template renderer
        self.helm_template_enabled = True

    def clone_repo(self, github_config: GitHubConfig) -> str:
        """Clone the GitHub repository and return the temporary directory path."""
        temp_dir = "/Users/sanjay.mogra/test_temp_dir"
        try:
            # Clone with authentication
            auth_repo_url = github_config.repo_url.replace('https://', f'https://{github_config.github_token}@')
            repo = Repo.clone_from(auth_repo_url, temp_dir, branch=github_config.github_branch)
            
            # Create new branch with timestamp
            new_branch = f"manifest-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            current = repo.create_head(new_branch)
            current.checkout()
            
            logger.info(f"Successfully cloned repository to {temp_dir} and created branch {new_branch}")
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir)
            logger.error(f"Failed to clone repository: {e}")
            raise

    def analyze_manifest(self, content: str, file_path: str) -> ManifestAnalysis:
        """Analyze a Kubernetes manifest content using AI for potential issues and improvements."""
        try:
            # Prepare manifest for AI analysis
            manifest_str = content
            #system_prompt = f"Fix this Kubernetes YAML by updating deprecated APIs, correcting syntax and values like wrong kind names, and return only the corrected YAML with no explanations, comments, or extra text. Below is the file content:\n{manifest_str}"
            system_prompt = f"""Fix this Kubernetes YAML: Update deprecated APIs, correct syntax errors, fix invalid resource types/values, and ensure proper formatting. Return ONLY the corrected YAML within a YAML code block (```yaml).: {manifest_str}"""
    
            #system_prompt = "You are a Kubernetes expert. Analyze the manifest for best practices, security issues, and potential improvements."

            if not self.use_ollama:
                # Use OpenAI for analysis
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": system_prompt}
                    ]
                )
                analysis = response.choices[0].message.content
            else:
                # Use Ollama for analysis
                try:
                    response = self.ollama_client.chat(
                        model='GandalfBaum/llama3.1-claude3.7',  # Using llama2 as default model
                        messages=[
                           # {"role": "system", "content": system_prompt},
                            {"role": "user", "content": system_prompt}
                        ]
                    )
                    analysis = response['message']['content']
                   
                     # Assuming analysis is the updated manifes
                    print(f"Ollama analysis: {analysis}")
                    # Extract only the YAML part (starting with a known key like "apiVersion")
                    #cleaned_output = re.sub(r"^```.*?$", "", analysis.strip(), flags=re.MULTILINE)
                    #cleaned_output = re.sub(r"^(Here is.*|Corrected YAML.*|Output:.*)$", "", cleaned_output, flags=re.IGNORECASE | re.MULTILINE).strip()
                    #cleaned_output = cleaned_output.strip()
                    #print(f"Cleaned output: {cleaned_output}")
                    #updated_manifest = yaml.safe_load(cleaned_output)
                    match = re.search(r"```yaml\s*\n(.*?)```", analysis, re.DOTALL)
                    updated_manifest = yaml.safe_load(match.group(1).strip())
                    print(f"Updated manifest: {updated_manifest}")
                except Exception as e:
                    logger.error(f"Error using Ollama model: {e}")
                    return ManifestAnalysis(manifest=updated_manifest, 
                    #issues=[str(e)], suggestions=[]
                    )

            # Process AI suggestions
            #issues = validation_errors + [issue.strip() for issue in analysis.split('\n') if issue.strip().startswith('- Issue:')]
            #suggestions = [sugg.strip() for sugg in analysis.split('\n') if sugg.strip().startswith('- Suggestion:')]

            return ManifestAnalysis(
                manifest=updated_manifest,
                #issues=issues, suggestions=suggestions
            )
        except Exception as e:
            logger.error(f"Error analyzing manifest {file_path}: {e}")
            return ManifestAnalysis(manifest=manifest, 
            #issues=[str(e)], suggestions=[]
            )

    def validate_manifest(self, manifest: Dict[str, Any]) -> List[str]:
        """Validate Kubernetes manifest against schema and best practices."""
        errors = []
        try:
            # Basic schema validation
            required_fields = ['apiVersion', 'kind', 'metadata']
            for field in required_fields:
                if field not in manifest:
                    errors.append(f"Missing required field: {field}")

            # Version compatibility check
            if 'apiVersion' in manifest:
                api_version = manifest['apiVersion']
                if api_version.startswith('apps/v1beta'):
                    errors.append(f"Deprecated API version: {api_version}. Use apps/v1 instead.")

            # Resource requirements check
            if manifest.get('kind') in ['Deployment', 'StatefulSet', 'DaemonSet']:
                containers = manifest.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                for container in containers:
                    if not container.get('resources'):
                        errors.append(f"Container {container.get('name')} missing resource limits/requests")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return errors

    def update_manifest_file(self, file_path: str, manifest: Dict[str, Any]) -> None:
        """Update the manifest file with the improved version."""
        try:
            with open(file_path, 'w') as f:
                yaml.safe_dump(manifest, f)
            logger.info(f"Successfully updated manifest file: {file_path}")
        except Exception as e:
            logger.error(f"Error updating manifest file {file_path}: {e}")
            raise

    def commit_and_push_changes(self, repo_path: str) -> None:
        """Commit and push changes to the repository."""
        try:
            repo = Repo(repo_path)
            # Add all changes
            repo.git.add(A=True)
            # Commit changes
            commit_message = f"Update Kubernetes manifests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            repo.index.commit(commit_message)
            # Get current branch name
            current_branch = repo.active_branch.name
            # Set up tracking and push
            repo.git.push('--set-upstream', 'origin', current_branch)
            logger.info("Successfully committed and pushed changes")
        except Exception as e:
            logger.error(f"Error committing and pushing changes: {e}")
            raise

    def process_manifests(self, repo_path: str, manifest_path: str, scan_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process all Kubernetes manifests in the specified directory.
        
        Args:
            repo_path: Base path of the cloned repository
            manifest_path: Default manifest directory name
            scan_path: Optional path to start scanning from. If None, starts from repo base
        """
        manifests = []
        manifest_dir = manifest_path if os.path.isabs(manifest_path) else os.path.join(repo_path, manifest_path)

        try:
            for root, _, files in os.walk(manifest_dir):
                for file in files:
                    if file.endswith(('.yaml', '.yml')):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Directly analyze the YAML content
                        try:
                            analysis = self.analyze_manifest(content, file_path)
                            if analysis.manifest:
                                # Update the file with the AI-suggested changes
                                self.update_manifest_file(file_path, analysis.manifest)
                                manifests.append(analysis.manifest)
                                logger.info(f"Successfully updated {file} with AI analysis")
                            else:
                                # If no updates, keep track of the original content
                                manifests.append(content)
                                logger.warning(f"No updates for {file}, using original content")
                            logger.info(f"Successfully processed {file}")
                        except Exception as e:
                            logger.error(f"Error processing {file}: {e}")
                            continue

            print(f"Manifests: {manifests}")
            return manifests
        except Exception as e:
            logger.error(f"Error processing manifests: {e}")
            raise


@app.post("/process-github", response_model=List[Dict[str, Any]])
async def process_github_repo(github_config: GitHubConfig):
    """Process Kubernetes manifests from a GitHub repository using configuration from request body.
    
    Example request body:
    {
        "repo_url": "https://github.com/username/repo",
        "github_token": "your-github-token",
        "github_branch": "main",
        "openai_api_key": "your-openai-api-key",
        "manifest_path": "manifests",
        "scan_path": null
    }
    """
    try:
        # Initialize MCP server with OpenAI API key from request
        server = MCPServer(github_config.openai_api_key)
        
        # Clone repository using provided configuration
        repo_path = server.clone_repo(github_config)
        try:
            # Process manifests
            manifests = server.process_manifests(repo_path, github_config.manifest_path, github_config.scan_path)
            # Commit and push changes
            server.commit_and_push_changes(repo_path)
            return manifests
        finally:
            # Cleanup temporary directory
            shutil.rmtree(repo_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)