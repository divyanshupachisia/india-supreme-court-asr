import csv
import os
import json
import pandas as pd
import typing as T

from datasets import Dataset, Audio
from youtube_transcript_api import YouTubeTranscriptApi

from data_utils.utils import extract_youtube_video_id
from data_utils.data_loader import DataLoader
from data_utils.pdf_to_text import PDFTranscriptProcessor
from data_utils.youtube_caption_aligner import YouTubeCaptionAligner
from data_utils.chunker import Chunker

CSV_FILE = "case_data.csv"
AUDIO_COLUMN_NAME = "mp3 format link"
TRANSCRIPT_COLUMN_NAME = "Transcript Link"
YOUTUBE_COLUMN_NAME = "Oral Hearing Link"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_FOLDER = os.path.join(CURRENT_DIR, "raw_data")
CHUNKED_DATA_FOLDER = os.path.join(CURRENT_DIR, "chunked_data")
CHUNK_LENGTH = 30  # [s]

PUSH_TO_HUGGING_FACE = True
HUGGING_FACE_REPO = "divi212/india-supreme-court-audio"


def _validate_csv_columns(csv_file: str) -> bool:
    """
    Validates the CSV file to check if it contains the required columns.

    :return: True if the CSV contains the required columns, False otherwise.
    """
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        return all(
            column in columns
            for column in [
                AUDIO_COLUMN_NAME,
                TRANSCRIPT_COLUMN_NAME,
                YOUTUBE_COLUMN_NAME,
            ]
        )


def _validate_metadata(row: T.Dict) -> bool:
    """
    Makes sure a row in a csv does not have inconsistencies and has valid data
    """
    try:
        YouTubeTranscriptApi.get_transcript(
            extract_youtube_video_id(row[YOUTUBE_COLUMN_NAME])
        )
    except Exception as e:
        print(f"An error occured while trying to load in youtube video: {e}")
        return False
    return True


def main():
    """
    Main loop to generate labeled data. Takes in a csv with link to an mp3 file, a corresponding
    youtube link, and a link to download the transcript from.
    Runs through 4 steps per row:
    (1) Download raw pdf transcript and mp3 audio file
    (2) Convert raw pdf transcript to text file, removing unnecessary information
    (3) Aligns transcript text to the audio file using an Aligner
    (4) Chunks the data into CHUNK_LENGTH

    """
    csv_valid = _validate_csv_columns(CSV_FILE)
    chunked_metadata = pd.DataFrame(columns=["audio", "transcript"])
    if not csv_valid:
        raise ValueError("CSV invalid!")
    with open(CSV_FILE, mode="r", encoding="utf-8") as file:
        os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
        os.makedirs(CHUNKED_DATA_FOLDER, exist_ok=True)
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            print(
                "----------------------------------------------------------------------------------"
            )
            print(f"Processing row {index}")
            base_folder = os.path.join(RAW_DATA_FOLDER, f"{index}")
            # validate and write metadata to a json file
            valid_row = _validate_metadata(row)
            if not valid_row:
                print(
                    "----------------------------------------------------------------------------------"
                )
                continue
            # get raw audio and transcript data
            print("Step 1 / 4: Downloading data from csv links")
            data_loader = DataLoader(
                base_folder, row[AUDIO_COLUMN_NAME], row[TRANSCRIPT_COLUMN_NAME]
            )
            audio_path, transcript_path = data_loader.load_data()
            if not audio_path or not transcript_path:
                print(
                    "Audio or transcript could not be loaded. Moving on to the next row."
                )
                continue
            # convert pdf to text
            print("Step 2 / 4: Extracting Text from the pdf transcript")
            pdf_processor = PDFTranscriptProcessor(transcript_path)
            transcript_text_path = pdf_processor.process()
            # use youtube caption aligner
            print(
                "Step 3 / 4: Adding timestamps to transcript through forced alignment."
            )
            youtube_aligner = YouTubeCaptionAligner(
                audio_path, transcript_text_path, row[YOUTUBE_COLUMN_NAME], base_folder
            )
            aligned_transcript_path = youtube_aligner.process()
            # chunk data
            if aligned_transcript_path:
                print("Step 4 / 4: Chunking data")
                chunked_data_folder = os.path.join(CHUNKED_DATA_FOLDER, f"{index}")
                chunker = Chunker(
                    aligned_transcript_path,
                    audio_path,
                    chunked_data_folder,
                    CHUNK_LENGTH,
                )
                row_metadata = chunker.process()
                chunked_metadata = pd.concat(
                    [chunked_metadata, pd.DataFrame(row_metadata)], ignore_index=True
                )
            print(
                "----------------------------------------------------------------------------------"
            )

    print(
        "Done! Writing overall mapping of chunked audio file to transcript and pushing to hugging face"
    )
    chunked_metadata.to_csv(
        f"{CHUNKED_DATA_FOLDER}/chunked_audio_metadata.csv", index=False
    )
    if PUSH_TO_HUGGING_FACE:
        print("Pushing data to hugging face")
        ds = Dataset.from_csv(f"{CHUNKED_DATA_FOLDER}/chunked_audio_metadata.csv")
        shuffled_ds = ds.shuffle(seed=212)
        train_test_ds = shuffled_ds.train_test_split(test_size=0.1, seed=212)
        train_test_ds = train_test_ds.cast_column("audio", Audio())
        train_test_ds.push_to_hub(HUGGING_FACE_REPO)


if __name__ == "__main__":
    main()
