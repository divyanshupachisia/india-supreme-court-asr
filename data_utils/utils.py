import re
import jiwer
import json
import nltk
import subprocess
import typing as T

from nltk.corpus import stopwords
from pydub import AudioSegment
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TranscriptChunk:
    """
    Stores a transcript along with corresponding start and end times in the audio file.
    """

    text: str = ""
    start: float = 0.0
    end: float = 0.0


def transcript_chunks_to_json(
    chunks: T.List[TranscriptChunk], out_path: str
) -> T.Optional[str]:
    """
    Takes in a list of transcript chunks and saves them to a JSON file specified by out_path.
    """
    try:
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump([asdict(chunk) for chunk in chunks], file, indent=4)
        return out_path
    except Exception as e:
        print(f"Error encountered while trying to write to {out_path}: {e}")
        return None


def json_to_transcript_chunks(json_path: str) -> T.List[TranscriptChunk]:
    """
    Reads the JSON file specified by json_path and returns a list of TranscriptChunk instances.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return [TranscriptChunk(**item) for item in data]
    except Exception as e:
        print(f"Error encountered while reading {json_path}: {e}")
        return []


def get_default_stop_words() -> set:
    """
    jbjk
    """
    # Check if the stopwords are already downloaded
    try:
        # Attempt to access the stopwords to see if they are available
        stop_words = stopwords.words("english")
    except LookupError:
        # If not available, download the data
        nltk.download("stopwords")
        stop_words = stopwords.words("english")

    return set(stop_words)


def extract_youtube_video_id(url: str) -> str:
    """Given a YouTube url, returns the video ID."""
    # TODO(divyanshu): This regular expression is brittle. Use a library to do this.
    pattern = r"(?:https?://)?(?:www\.)?youtube\.com/(?:watch\?v=|live/)([^&\n?#]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid YouTube URL: {url}")


# NOTE(divyanshu): This was needed because AudioSegment didn't play nice with mp3 files. Also on
# vscode wave files can be played, so this format is overall easier to work with.
def mp3_to_wav(input_mp3_path, output_wav_path=None) -> str:
    """
    Converts an MP3 file to a WAV file using ffmpeg.

    Args:
    input_mp3_path (str): Path to the input MP3 file.
    output_wav_path (str): Optional; Path to save the output WAV file. If not provided, it replaces
    the extension of the input path.

    Returns:
    str: Path to the converted WAV file.
    """
    if output_wav_path is None:
        output_wav_path = input_mp3_path.rsplit(".", 1)[0] + ".wav"
    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-i",
        input_mp3_path,  # Input file
        "-acodec",
        "pcm_s16le",  # Convert audio to PCM 16-bit little-endian
        "-ar",
        "44100",  # Set sample rate to 44100 Hz
        "-ac",
        "2",  # Set audio channels to stereo
        output_wav_path,  # Output file
    ]
    # Execute the ffmpeg command
    try:
        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        return output_wav_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"ffmpeg error: {e.stderr.decode()}") from e


def get_audio_length(audio_file_path: str) -> float:
    """Returns the length of the audio in seconds"""
    try:
        audio = AudioSegment.from_file(audio_file_path)
        # Convert from ms to s
        duration_in_seconds = len(audio) / 1000.0
        return duration_in_seconds
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0.0


def compute_wer(reference: str, comparison: str) -> float:
    """Word Error Rate (WER) for two strings."""
    return jiwer.wer(reference, comparison)


def compute_cosine_similarity(reference: str, comparison: str) -> float:
    """
    Uses TF-IDF to compute the cosine similarity between two strings. This is meant to signify how
    similar in meaning the two strings are.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([reference, comparison])
    return cosine_similarity(X[0], X[1])


def find_sublist_in_list(sublist: T.List, lst: T.List):
    """Finds sublist in list."""
    for i in range(len(lst) - len(sublist) + 1):
        if lst[i : i + len(sublist)] == sublist:
            return i
    return -1
