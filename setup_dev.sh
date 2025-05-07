#!/bin/bash
# Setup script for DeepWiki development environment

echo "🚀 Setting up DeepWiki development environment..."

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed or not in PATH. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Using Python: $($PYTHON_CMD --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    if [ ! -d "venv" ]; then
        echo "❌ Failed to create virtual environment. Installing virtualenv and trying again..."
        $PYTHON_CMD -m pip install virtualenv
        $PYTHON_CMD -m virtualenv venv
        
        if [ ! -d "venv" ]; then
            echo "❌ Virtual environment creation failed. Please create it manually with:"
            echo "$PYTHON_CMD -m venv venv"
            exit 1
        fi
    fi
else
    echo "📦 Virtual environment already exists."
fi

# Set the absolute path to the virtual environment
VENV_PATH="$(pwd)/venv"
echo "📁 Virtual environment path: $VENV_PATH"

# Determine the activation script based on OS
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS or Linux
    ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    ACTIVATE_SCRIPT="$VENV_PATH/Scripts/activate"
else
    echo "❌ Unsupported OS. Please activate the virtual environment manually."
    exit 1
fi

# Check if activation script exists
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "❌ Activation script not found at: $ACTIVATE_SCRIPT"
    echo "Virtual environment may be corrupted. Try removing the venv directory and running this script again."
    exit 1
fi

echo "🔄 Activating virtual environment from: $ACTIVATE_SCRIPT"
source "$ACTIVATE_SCRIPT"

# Verify the virtualenv is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Failed to activate virtual environment."
    echo "Please activate it manually:"
    echo "source $ACTIVATE_SCRIPT"
    exit 1
fi

echo "✅ Virtual environment activated."

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r api/requirements.txt

# Ensure the ChromaDB persistence directory exists
mkdir -p ~/.deepwiki/chromadb

# Check for API keys
MISSING_KEYS=false

if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ] && grep -q "OPENAI_API_KEY" .env; then
        echo "✅ OPENAI_API_KEY found in .env file."
    else
        echo "⚠️ OPENAI_API_KEY not found in environment or .env file."
        MISSING_KEYS=true
    fi
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    if [ -f ".env" ] && grep -q "GOOGLE_API_KEY" .env; then
        echo "✅ GOOGLE_API_KEY found in .env file."
    else
        echo "⚠️ GOOGLE_API_KEY not found in environment or .env file."
        MISSING_KEYS=true
    fi
fi

if [ "$MISSING_KEYS" = true ]; then
    echo ""
    echo "⚠️ Some API keys are missing. Create a .env file with:"
    echo "OPENAI_API_KEY=your_openai_key"
    echo "GOOGLE_API_KEY=your_google_key"
    echo ""
fi

echo ""
echo "✅ Setup complete! The development environment is ready."
echo ""
echo "To start using the LangGraph RAG pipeline, first activate the virtual environment (if running in a new terminal):"
echo "source $(pwd)/venv/bin/activate"
echo ""
echo "Then run the test script:"
echo "python -m api.test_langgraph --repo https://github.com/username/repo"
echo "  or"
echo "python -m api.test_langgraph --local /path/to/local/directory"
echo ""
echo "Happy coding! 🎉" 