#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import sys

def deploy_to_pi():
    # Configuration
    PI_HOST = "pi.local"
    PI_USER = "dev"
    REMOTE_DIR = "~/"
    
    try:
        # Use rsync to copy files
        print("Syncing files to Raspberry Pi...")
        rsync_command = [
            'rsync',
            '-av',                     # archive mode, verbose
            '--progress',              # show progress
            '--exclude', '.git/',      # exclude git directory
            '--exclude', 'venv/',      # exclude virtual environment
            '--exclude', '__pycache__/',
            '--exclude', '*.pyc',
            '--exclude', '*.pyo',
            '--exclude', '*.pyd',
            '--exclude', '.DS_Store',
            '--include', '.env',
            './',                      # current directory
            f'{PI_USER}@{PI_HOST}:{REMOTE_DIR}/'
        ]
        
        # If you have a .gitignore file, use it for excludes
        if os.path.exists('.gitignore'):
            rsync_command.insert(-2, '--exclude-from=.gitignore')
            
        subprocess.run(rsync_command, check=True)
        print("\nFiles deployed successfully to Raspberry Pi")
        
    except subprocess.CalledProcessError as e:
        print(f"Error deploying files: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    deploy_to_pi()
