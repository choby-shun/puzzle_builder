#!/bin/bash

PROJECT_NAME="."

mkdir -p $PROJECT_NAME/{bot/{handlers,puzzles},tests}

touch $PROJECT_NAME/.env
touch $PROJECT_NAME/requirements.txt
touch $PROJECT_NAME/README.md
touch $PROJECT_NAME/main.py

# Create __init__.py files
touch $PROJECT_NAME/bot/__init__.py
touch $PROJECT_NAME/bot/handlers/__init__.py
touch $PROJECT_NAME/bot/puzzles/__init__.py
touch $PROJECT_NAME/tests/__init__.py

# Create handler stubs
cat > $PROJECT_NAME/bot/handlers/start.py <<EOF
from telegram import Update
from telegram.ext import ContextTypes

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the Puzzle Bot!")
EOF

cat > $PROJECT_NAME/bot/handlers/help.py <<EOF
from telegram import Update
from telegram.ext import ContextTypes

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Use /start to begin the puzzle!")
EOF

cat > $PROJECT_NAME/bot/handlers/puzzle.py <<EOF
from telegram import Update
from telegram.ext import ContextTypes

async def puzzle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implement puzzle logic here
    await update.message.reply_text("Here's your puzzle...")
EOF

# Create puzzle stubs
cat > $PROJECT_NAME/bot/puzzles/puzzle1.py <<EOF
def get_puzzle():
    return "What has keys but can't open locks?"
EOF

cat > $PROJECT_NAME/bot/puzzles/puzzle2.py <<EOF
def get_puzzle():
    return "What comes once in a minute, twice in a moment, but never in a thousand years?"
EOF

# Create states and utils
cat > $PROJECT_NAME/bot/states.py <<EOF
# Simple in-memory state (extendable to DB or Redis)
user_states = {}

def set_state(user_id, state):
    user_states[user_id] = state

def get_state(user_id):
    return user_states.get(user_id)
EOF

cat > $PROJECT_NAME/bot/utils.py <<EOF
def log(message):
    print(f"[LOG] {message}")
EOF

cat > $PROJECT_NAME/bot/config.py <<EOF
import os

BOT_TOKEN = os.getenv("BOT_TOKEN", "your-token-here")
