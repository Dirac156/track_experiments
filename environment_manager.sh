#!/bin/bash

# environment_manager.sh: Script to manage Python virtual environments and requirements.
# Usage:
#   - Initialize: ./environment_manager.sh init <env_name>
#   - Activate: source ./environment_manager.sh activate <env_name>
#   - Close: ./environment_manager.sh close <env_name>

ENV_NAME=$2  # Name of the virtual environment

function init_environment() {
    echo "Creating a virtual environment named '$ENV_NAME'..."
    python -m venv $ENV_NAME

    if [[ -d "$ENV_NAME" ]]; then
        echo "Virtual environment '$ENV_NAME' created successfully."
    else
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
}

function activate_environment() {
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        source $ENV_NAME/Scripts/activate
    else
        source $ENV_NAME/bin/activate
    fi
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Virtual environment '$ENV_NAME' activated."
    else
        echo "Error: Failed to activate virtual environment."
        exit 1
    fi
}

function close_environment() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Freezing requirements to requirements.txt..."
        pip freeze > requirements.txt
        echo "Deactivating virtual environment..."
        deactivate
        echo "Environment closed and requirements saved."
    else
        echo "Error: No active virtual environment to close."
        exit 1
    fi
}

case $1 in
    init)
        init_environment
        ;;
    activate)
        activate_environment
        ;;
    close)
        close_environment
        ;;
    *)
        echo "Invalid command. Usage: ./environment_manager.sh [init|activate|close] <env_name>"
        exit 1
        ;;
esac
