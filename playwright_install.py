import subprocess
import os
import streamlit as st

# Function to ensure Playwright browsers are installed
def install_playwright_browsers():
    # Check if browsers are already installed
    playwright_dir = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(playwright_dir) or not os.listdir(playwright_dir):
        st.write("Installing Playwright browsers...")
        try:
            # Run the install command
            subprocess.run(["playwright", "install"], check=True)
            st.write("Playwright browsers installed successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install Playwright browsers: {e}")
    else:
        st.write("Playwright browsers already installed.")