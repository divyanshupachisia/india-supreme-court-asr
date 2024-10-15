import typing as T
from .utils import TranscriptChunk, transcript_chunks_to_json, get_audio_length


class Aligner:
    """
    Base class to align text from a transcript source without timestamps to its audio file.
    """

    def __init__(
        self,
        audio_source: str,
        transcript_source: str,
        save_folder: T.Optional[str] = None,
    ):
        self.audio_source = audio_source
        self.transcript_source = transcript_source
        self.save_folder = save_folder

    def process(self) -> T.Optional[str]:
        """
        Will perform alignment with the audio and transcript source and save the aligned transcript
        to the specified path
        """
        self._fetch_data()
        if not self._sanity_check_data():
            print("It is likely the audio and text provided do not match, skipping.")
            return None
        results = self._process_alignment()
        out_path = None
        if self.save_folder:
            out_path = self._save_results(results)
        return out_path

    def _get_audio_souce_length(self) -> float:
        """
        Returns audio length in seconds
        """
        return get_audio_length(self.audio_source)

    def _fetch_data(self) -> None:
        """Method to fetch the data needed for alignment. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _sanity_check_data(self) -> bool:
        """Method to sanity check the audio/text align. To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _process_alignment(self) -> T.List[TranscriptChunk]:
        """Process the alignment according to the specific aligner's algorithm. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _save_results(self, results: T.List[TranscriptChunk]) -> T.Optional[str]:
        """Save the results of the alignment to a JSON file."""
        if not self.save_folder:
            raise ValueError("Save path must be provided to save results.")
        path = transcript_chunks_to_json(
            results, f"{self.save_folder}/aligned_transcript.json"
        )
        return path
