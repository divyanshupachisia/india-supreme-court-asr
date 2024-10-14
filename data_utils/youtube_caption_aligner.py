import typing as T
import string

from youtube_transcript_api import YouTubeTranscriptApi

from .aligner import Aligner
from .utils import TranscriptChunk
from .utils import (
    extract_youtube_video_id,
    get_default_stop_words,
    transcript_chunks_to_json,
)


class YouTubeCaptionAligner(Aligner):
    """
    Takes in a link to a youtube video, using the youtube transcript API to generates timestamped
    caption.
    These timestamped captions are then matched with the transcript, which is then outputed.
    """

    def __init__(
        self,
        video_url: str,
        transcript_path: str,
        save_folder: T.Optional[str] = None,
        unmatchable_words: T.Optional[set] = None,
    ):
        super().__init__(
            audio_source=video_url,
            transcript_source=transcript_path,
            save_folder=save_folder,
        )
        self.video_id = extract_youtube_video_id(video_url)
        self.unmatchable_words = (
            unmatchable_words if unmatchable_words else get_default_stop_words()
        )
        self._transcript_list: T.List[str] = []
        self._youtube_caption: T.List[TranscriptChunk] = []

    def _fetch_data(self) -> None:
        try:
            for chunk in YouTubeTranscriptApi.get_transcript(self.video_id):
                self._youtube_caption.append(
                    TranscriptChunk(
                        text=chunk["text"],
                        start=chunk["start"],
                        end=chunk["start"] + chunk["duration"],
                    )
                )
            transcript_chunks_to_json(
                self._youtube_caption, f"{self.save_folder}/youtube_captions.json"
            )
            with open(self.transcript_source, "r", encoding="utf-8") as file:
                self._transcript_list = file.read().split()
        except Exception as e:
            raise ValueError(f"Could not fetch data from youtube: {e}")

    def _process_alignment(self) -> T.List[TranscriptChunk]:
        if not self._youtube_caption or not self._transcript_list:
            raise ValueError("Data for alignment is missing.")
        return self._align_captions()

    def _validate_youtube_caption(self, caption: TranscriptChunk) -> bool:
        text = caption.text
        if len(text.split()) == 1:
            return False
        if text.lower() in self.unmatchable_words:
            return False
        return True

    def _align_captions(self) -> T.List[TranscriptChunk]:
        start_index = 0
        len_cur_text = 0
        aligned_transcripts: T.List[TranscriptChunk] = []

        for i in range(0, len(self._youtube_caption) - 1):

            cur_caption = self._youtube_caption[i]
            next_caption = self._youtube_caption[i + 1]

            if not (self._validate_youtube_caption(next_caption)):
                continue

            len_cur_text += len(cur_caption.text.split())
            # TODO(divyanshu): perhaps make this a setting to the aligner
            buffer = len_cur_text

            next_text = self._youtube_caption[i + 1].text
            word_to_match = next_text.split()[0]

            search_range = self._transcript_list[
                start_index
                + len_cur_text
                - buffer : min(
                    start_index + len_cur_text + buffer, len(self._transcript_list)
                )
            ]
            center_index = len(search_range) // 2
            matches = [
                index
                for index, match in enumerate(search_range)
                if self._remove_punctuation(match.lower())
                == self._remove_punctuation(word_to_match.lower())
            ]

            if matches:
                # NOTE(divyanshu): If there are too many matches, just move on so we can find a more
                # unique word that will help us localize better
                # TODO(divyanshu): perhaps make this a setting to the aligner
                if len(matches) > 2:
                    continue
                start_time = cur_caption.start
                end_time = next_caption.start
                closest_match_index = min(matches, key=lambda x: abs(x - center_index))
                aligned_transcripts.append(
                    TranscriptChunk(
                        text=" ".join(search_range[:closest_match_index]),
                        start=start_time,
                        end=end_time,
                    )
                )
                start_index += len_cur_text - buffer + closest_match_index
                len_cur_text = 0

        last_aligned_transcript = aligned_transcripts[-1]
        last_youtube_caption = self._youtube_caption[-1]
        aligned_transcripts.append(
            TranscriptChunk(
                text=" ".join(self._transcript_list[start_index:]),
                start=last_aligned_transcript.end,
                end=last_youtube_caption.end,
            )
        )
        return aligned_transcripts

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))
