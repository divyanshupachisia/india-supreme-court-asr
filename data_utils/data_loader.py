import os
import typing as T
import requests
import urllib3
from tqdm import tqdm
from .utils import mp3_to_wav


class DataLoader:
    """
    Takes in an audio and transcript download URL and downloads and saves them to the specified
    folder.
    """

    def __init__(
        self,
        save_folder: str,
        audio_download_url: str,
        transcript_download_url: str,
    ) -> None:
        """
        Initializes the DataLoader with the file path, save path, and columns that contain links for
        audio and transcripts.

        :param save_folder: Directory path where downloaded files should be saved.
        :param audio_download_url: Column containing URLs to audio files.
        :param transcript_download_url: Column containing URLs to transcript files.
        """
        self.save_folder = save_folder
        self.audio_download_url = audio_download_url
        self.transcript_download_url = transcript_download_url

    def load_data(self) -> T.Tuple[T.Optional[str], T.Optional[str]]:
        """
        Loads data from the CSV file, validates its structure, and downloads files from URLs.

        :return: A list of dictionaries, each representing a row from the CSV with downloaded file paths.
        """
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        if not self.audio_download_url or not self.transcript_download_url:
            raise ValueError("Audio and transcript URL not specified")

        mp3_file_path = self.download_file(
            self.audio_download_url, f"{self.save_folder}/audio.mp3"
        )
        wav_file_path = mp3_to_wav(mp3_file_path, f"{self.save_folder}/audio.wav")

        transcript_file_path = self.download_file(
            self.transcript_download_url, f"{self.save_folder}/transcript.pdf"
        )

        return wav_file_path, transcript_file_path

    def download_file(self, url: str, file_name: str) -> T.Optional[str]:
        """
        Downloads a file from a specified URL and saves it with a given file name.
        Includes a progress bar to show download progress.

        :param url: The URL to download the file from.
        :param file_name: The name under which the file will be saved.
        :return: The full path to the saved file.
        """
        url = self._sanitize_url(url)
        # NOTE(divyanshu): main.sci.gov has an expired SSL certificate. Disable warnings to prevent
        # spam.
        if "main.sci.gov" in url:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            response = requests.get(url, stream=True, timeout=100, verify=False)
        else:
            response = requests.get(url, stream=True, timeout=100)

        if response.status_code == 200:
            print(f"Downloading {file_name} from {url}...")

            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 8192  # Set the block size to a reasonable value

            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(file_name, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

            progress_bar.close()
            return file_name
        else:
            print(f"Request failed with status code: {response.status_code}")
            return None

    def _sanitize_url(self, url: str) -> str:
        """Make modifications to the URL if needed"""
        # Check if the URL is a Dropbox link and modify it for direct download
        if "dropbox.com" in url:
            if "dl=0" in url:
                url = url.replace("dl=0", "dl=1")
            elif "dl=1" not in url:
                url += "&dl=1" if "?" in url else "?dl=1"

        return url
