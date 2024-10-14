import fitz
import re


class PDFTranscriptProcessor:
    """
    Takes in a PDF transcript, and outputs a text file with only the transcript.
    """

    def __init__(
        self,
        file_path: str,
        ignore_first_page: bool = True,
        remove_speaker_names: bool = True,
    ):
        self.file_path = file_path
        self.ignore_first_page = ignore_first_page
        self.remove_speaker_names = remove_speaker_names

    def process(self) -> str:
        """
        Main loop
        """
        text = self._convert_pdf_to_text()
        clean_text = self._sanitize_text(text)
        # Replace the file extension from .pdf to .txt
        txt_file_path = self.file_path.rsplit(".", 1)[0] + ".txt"
        with open(txt_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(clean_text)
        return txt_file_path

    def _convert_pdf_to_text(self) -> str:
        """
        Get text from a PDF and optionally save it to a text file.

        Args:
        file_path (str): The path to the PDF file.
        ignore_first_page (bool): Whether to ignore the first page of the PDF.
        save_file (bool): Whether to save the extracted text to a .txt file.

        Returns:
        str: The extracted text from the PDF.
        """
        doc = fitz.open(self.file_path)
        text = ""
        start_index = 1 if self.ignore_first_page else 0
        for page in doc[start_index:]:
            text += page.get_text()
        return text

    def _sanitize_text(self, text: str) -> str:
        """ """
        text = self._remove_line_numbers(text)
        text = text.replace("Transcribed by TERES", "")
        text = self._remove_timestamps(text)
        text = text.replace("END OF THIS PROCEEDING", "")
        text = text.replace("END OF DAY * PROCEEDINGS", "")
        if self.remove_speaker_names:
            text = self._remove_speaker_names(text)
        return text

    def _remove_line_numbers(self, text: str) -> str:
        """Removes line numbers from the start of each line in the provided text."""
        return re.sub(r"^\s*\d+\s+", "", text, flags=re.MULTILINE)

    def _remove_timestamps(self, text: str) -> str:
        """
        Removes timestamps from the provided text.
        Assumes timestamps contain IST at the end.
        """
        timestamp_pattern = r"\d{1,2}:\d{2}\s*[AP]M\s*IST"
        return re.sub(timestamp_pattern, "", text)

    def _remove_speaker_names(self, text: str) -> str:
        """
        Removes speaker names from the provided text.
        Assumes speaker names are in ALL_CAPS followed by a ":"
        """
        return re.sub(r"^.*?:\s*", "", text, flags=re.MULTILINE)

    def _remove_end_proceedings(self, text: str) -> str:
        """Removes the phase END OF * PROCEEDINGS?"""
        end_of_proceeding_pattern = r"END OF .*? PROCEEDINGS?"
        return re.sub(end_of_proceeding_pattern, "", text)
