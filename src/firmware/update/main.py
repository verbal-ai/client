import subprocess
import time
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGitUpdateService:
    def __init__(self, repo_path, repo_url, branch='main'):
        self.repo_path = repo_path
        self.repo_url = repo_url
        self.branch = branch
        self.commit_file = os.path.join(os.path.dirname(repo_path), '.last_commit')
        self.current_commit = self.load_saved_commit() or self.get_remote_commit()

    def get_remote_commit(self):
        """Get the latest commit hash from remote without cloning"""
        try:
            result = subprocess.run(
                ['git', 'ls-remote', self.repo_url, f'refs/heads/{self.branch}'],
                capture_output=True,
                text=True
            )
            return result.stdout.split()[0] if result.stdout else None
        except Exception as e:
            logger.error(f"Failed to get remote commit: {e}")
            return None

    def load_saved_commit(self):
        """Load the last known commit hash from file"""
        try:
            if os.path.exists(self.commit_file):
                with open(self.commit_file, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load saved commit: {e}")
        return None

    def save_commit(self, commit_hash):
        """Save the current commit hash to file"""
        try:
            os.makedirs(os.path.dirname(self.commit_file), exist_ok=True)
            with open(self.commit_file, 'w') as f:
                f.write(commit_hash)
        except Exception as e:
            logger.error(f"Failed to save commit: {e}")

    def check_for_updates(self):
        """Check if there are any updates in the remote repository"""
        try:
            new_commit = self.get_remote_commit()
            if new_commit and new_commit != self.current_commit:
                logger.info(f"New commit found: {new_commit}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return False

    def sync_and_update(self):
        """Perform update using rsync to efficiently update changed files"""
        try:
            # Create a temporary directory for the new clone
            temp_path = f"{self.repo_path}_temp"
            
            # Perform a shallow clone to the temporary directory
            clone_result = subprocess.run([
                'git', 'clone',
                '--depth', '1',
                '--single-branch',
                '--branch', self.branch,
                self.repo_url,
                temp_path
            ], capture_output=True, text=True)

            if clone_result.returncode != 0:
                raise Exception(f"Clone failed: {clone_result.stderr}")

            # Rsync the files from temp to main directory
            # -a: archive mode (preserves permissions, etc.)
            # -v: verbose
            # -z: compress
            # --delete: remove files that don't exist in source
            # --exclude: exclude .git directory to avoid conflicts
            rsync_result = subprocess.run([
                'rsync',
                '-avz',
                '--delete',
                '--exclude', '.git/',
                f"{temp_path}/",  # Source with trailing slash to copy contents
                f"{self.repo_path}/"  # Destination
            ], capture_output=True, text=True)

            if rsync_result.returncode != 0:
                raise Exception(f"Rsync failed: {rsync_result.stderr}")

            # Clean up temp directory
            subprocess.run(['rm', '-rf', temp_path])

            # Update pip requirements if needed
            self.update_requirements()
            
            # Restart the service
            self.restart_service()
            
            # Update the current commit hash and save it
            self.current_commit = self.get_remote_commit()
            self.save_commit(self.current_commit)
            
            logger.info(f"Update successful at {datetime.now()}")
            logger.debug(f"Rsync output: {rsync_result.stdout}")
            return True
        except Exception as e:
            logger.error(f"Update failed: {e}")
            # Clean up temp directory if something went wrong
            if os.path.exists(f"{self.repo_path}_temp"):
                subprocess.run(['rm', '-rf', f"{self.repo_path}_temp"])
            return False

    def update_requirements(self):
        """Update pip requirements"""
        requirements_file = os.path.join(self.repo_path, 'requirements-pi.txt')
        if os.path.exists(requirements_file):
            try:
                subprocess.run([
                    'pip3', 'install', '-r', requirements_file
                ], check=True)
                logger.info("Requirements updated successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to update requirements: {e}")

    def restart_service(self):
        """Restart the main service"""
        service_name = "your_service_name"  # Replace with your service name
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', service_name], check=True)
            logger.info(f"Service {service_name} restarted successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart service: {e}")

def main():
    REPO_URL = "https://github.com/verbal-ai/client"  # Replace with your repo URL
    REPO_PATH = "~/client"  # Replace with your repo path
    
    update_service = LightGitUpdateService(REPO_PATH, REPO_URL)
    
    while True:
        try:
            if update_service.check_for_updates():
                logger.info("Updates found, applying changes...")
                update_service.sync_and_update()
            time.sleep(300)  # Check every 5 minutes
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)  # Wait a minute before retrying if there's an error

if __name__ == "__main__":
    main()