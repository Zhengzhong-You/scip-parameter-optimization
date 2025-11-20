#!/bin/bash

# Git push script for staging specific files and pushing to remote repo
git add src .gitignore README.md install.py requirements.txt gitpush.sh
git commit -m "update test"
git push