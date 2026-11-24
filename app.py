import sys
from pathlib import Path

# Run the core streamlit application
src_path = Path(__file__).resolve().parent / "src" / "streamlit.py"
if __name__ == "__main__":
    import os
    os.system(f"streamlit run {src_path}")
