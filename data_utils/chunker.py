import json
import os
import typing as T
from pydub import AudioSegment
from .utils import TranscriptChunk, json_to_transcript_chunks


class Chunker:
    """
    Chunks into audio segments of specified chunk_length, and matches with the transcript.
    Produces data that can be fed into a standard ASR model.
    """

    def __init__(
        self,
        transcript_path: str,
        audio_path: str,
        save_directory: str,
        chunk_length: int,
    ):
        self.transcript: T.List[TranscriptChunk] = json_to_transcript_chunks(
            transcript_path
        )
        self.audio: AudioSegment = AudioSegment.from_wav(audio_path)
        self.save_directory: str = save_directory
        self.chunk_length: int = chunk_length
        os.makedirs(self.save_directory, exist_ok=True)

    def _get_chunks(self) -> T.List[TranscriptChunk]:
        """
        Returns a chunked transcript that are all as close as possible to self.chunk_length
        """
        first_segment = self.transcript[0]
        chunk_start = first_segment.start
        chunk_end = first_segment.end
        if chunk_end - chunk_start >= self.chunk_length:
            chunk_end = chunk_start + self.chunk_length
        chunk_text = first_segment.text

        chunks: T.List[TranscriptChunk] = []

        for segment in self.transcript[1:]:
            current_length = chunk_end - chunk_start
            addition_length = segment.end - chunk_end
            if (current_length + addition_length) >= self.chunk_length:
                chunks.append(TranscriptChunk(chunk_text, chunk_start, chunk_end))
                chunk_start = segment.start
                chunk_end = segment.end
                if chunk_end - chunk_start >= self.chunk_length:
                    chunk_end = chunk_start + self.chunk_length
                chunk_text = segment.text
            else:
                chunk_end = segment.end
                chunk_text += " " + segment.text
        chunks.append(TranscriptChunk(chunk_text, chunk_start, chunk_end))

        return chunks

    def process(self) -> T.List[T.Dict]:
        """
        Find the chunks, then split the audio accorindly and save.
        """
        chunks = self._get_chunks()
        results_dict: T.List[T.Dict] = []
        for chunk in chunks:
            chunk_filename: str = f"{chunk.start}_{chunk.end}.wav"
            # need to convert to ms when indexing audio
            chunk_audio: AudioSegment = self.audio[
                chunk.start * 1000 : chunk.end * 1000
            ]
            chunk_filepath = os.path.join(self.save_directory, chunk_filename)
            chunk_audio.export(chunk_filepath, format="wav")
            results_dict.append(
                {"audio": f"{chunk_filepath}", "transcript": chunk.text}
            )

        results_path: str = os.path.join(self.save_directory, "results.json")
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(results_dict, file, indent=4)

        return results_dict
