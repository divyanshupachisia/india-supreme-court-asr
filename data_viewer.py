import streamlit as st
import pandas as pd
from pathlib import Path
import random


# Load the CSV file
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


# Path to the CSV file
csv_file_path = "chunked_data/chunked_audio_metadata.csv"

if not Path(csv_file_path).exists():
    st.error(
        f"Error: The file '{csv_file_path}' does not exist. Run the data generation pipeline first."
    )
else:
    # Load data
    data = load_data(csv_file_path)

    # Title of the app
    st.title("Audio and Transcript Viewer")

    # Session state to keep track of selected index
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = 0

    # Sidebar for selecting an entry
    selected_index = st.sidebar.selectbox(
        "Select an entry", data.index, index=st.session_state.selected_index
    )

    # Update the session state when the selectbox is used
    st.session_state.selected_index = selected_index

    # Shuffle button
    if st.sidebar.button("Shuffle"):
        # This update triggers a rerun, and the selectbox index is updated automatically
        st.session_state.selected_index = random.randint(0, len(data) - 1)

    # Display the audio and transcript
    st.write("## Audio")
    audio_file = data.iloc[st.session_state.selected_index]["audio"]
    audio_path = Path(audio_file)
    if audio_path.exists():
        st.audio(str(audio_path))

    st.write("## Transcript")
    transcript = data.iloc[st.session_state.selected_index]["transcript"]
    st.write(transcript)
