import typing as T
import string

from youtube_transcript_api import YouTubeTranscriptApi

from .aligner import Aligner
from .utils import TranscriptChunk
from .utils import (
    compute_cosine_similarity,
    compute_wer,
    extract_youtube_video_id,
    find_sublist_in_list,
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
        audio_source: str,
        transcript_source: str,
        youtube_url: str,
        save_folder: T.Optional[str] = None,
        unmatchable_words: T.Optional[set] = None,
    ):
        super().__init__(
            audio_source=audio_source,
            transcript_source=transcript_source,
            save_folder=save_folder,
        )
        self.video_url = youtube_url
        self.unmatchable_words = (
            unmatchable_words if unmatchable_words else get_default_stop_words()
        )
        self._transcript_text: str = ""
        self._transcript_list: T.List[str] = []
        self._word_counts: T.Dict[str, int] = {}
        self._youtube_caption: T.List[TranscriptChunk] = []

    def _fetch_data(self) -> None:
        try:
            video_id = extract_youtube_video_id(self.video_url)
            for chunk in YouTubeTranscriptApi.get_transcript(video_id):
                self._youtube_caption.append(
                    TranscriptChunk(
                        text=chunk["text"],
                        start=chunk["start"],
                        end=chunk["start"] + chunk["duration"],
                    )
                )
            if self.save_folder:
                transcript_chunks_to_json(
                    self._youtube_caption, f"{self.save_folder}/youtube_captions.json"
                )
            with open(self.transcript_source, "r", encoding="utf-8") as file:
                self._transcript_text = file.read()
            self._transcript_list = self._transcript_text.split()
            self._word_counts = self._compute_word_counts_in_transcript()
        except Exception as e:
            raise ValueError(f"Could not fetch data from youtube: {e}")

    def _sanity_check_data(self) -> bool:
        """
        Makes sure we have paired the correct audio with text
        """
        # check that the length of the audio makes sense
        audio_source_length = self._get_audio_souce_length()  # [s]
        last_caption_end_time = self._youtube_caption[-1].end  # [s]

        if last_caption_end_time > audio_source_length:
            print(
                f"youtube caption has an end time: {last_caption_end_time} higher then the audio length: {audio_source_length}"
            )
            return False

        # check on the similarity of words between the youtube caption and transcript text
        reference_str = ""
        for caption in self._youtube_caption:
            reference_str += " " + self._remove_punctuation(caption.text.lower())
        transcript_str = self._remove_punctuation(self._transcript_text.lower())
        wer = compute_wer(reference_str, transcript_str)
        if wer > 0.5:
            print(
                f"Word error rate between youtube caption and transcript is higher than acceptable: {wer} > 0.5"
            )
            return False
        cosine_similarity = compute_cosine_similarity(reference_str, transcript_str)
        if cosine_similarity < 0.9:
            print(
                f"Cosine similarity between youtube caption and transcript is higher than acceptable: {cosine_similarity} > 0.9"
            )
            return False

        # Make sure the lengths of both texts are not very mismatched
        if not self._align_youtube_caption_transcript_starting_points():
            print(
                "Lengths of audio and text and mismatched and a match couldn't be made"
            )
            return False

        return True

    def _compute_word_counts_in_transcript(self) -> T.Dict[str, int]:
        word_counts = {}
        for word in self._transcript_list:
            formatted_word = self._remove_punctuation(word.lower())
            if formatted_word in word_counts:
                word_counts[formatted_word] += 1
            else:
                word_counts[formatted_word] = 1
        return word_counts

    def _process_alignment(self) -> T.List[TranscriptChunk]:
        if not self._youtube_caption or not self._transcript_list:
            raise ValueError("Data for alignment is missing.")
        return self._align_captions()

    def _validate_youtube_caption(self, caption: TranscriptChunk) -> bool:
        text = caption.text
        if len(text.split()) == 1:
            return False
        # match the first word since youtube has the time at which this was said and you don't have
        # to make assumptions of speaking rate
        word_to_match = caption.text.split()[0]
        # word doesn't appear in the transcript
        if word_to_match not in self._word_counts:
            return False
        # in words that we don't want to match against since they are too common
        if word_to_match in self.unmatchable_words:
            return False
        return True

    def _align_youtube_caption_transcript_starting_points(self) -> bool:
        """
        Aligns self._youtube_caption and self._transcript by removing entries until they both have
        the beginning match
        Returns True if it managed to align them and False if not.
        """
        # Get ratios of length as a heuristic to determine the order in which to search
        transcript_length = len(self._transcript_list)
        youtube_caption_length = sum(
            len(caption.text.split()) for caption in self._youtube_caption
        )
        transcript_to_youtube_ratio = transcript_length / youtube_caption_length

        # case 1: they are similar size so do nothing. The general matching algorithm should be able
        # to find the matches.
        if 0.9 < transcript_to_youtube_ratio < 1.1:
            return True

        # to store the indices for the actual alignment
        transcript_index = 0
        youtube_caption_index = 0
        alignment_found = False

        # case 2: transcript length is larger or similar size to youtube caption length. This means
        # that the transcript started before the video. So look through the beginning entries of the
        # youtube caption and find the transcript length
        if transcript_to_youtube_ratio > 1.1:
            transcript_text_list = self._remove_punctuation(
                self._transcript_text.lower()
            ).split()
            for i, caption in enumerate(self._youtube_caption[0:100]):
                caption_text_list = self._remove_punctuation(
                    caption.text.lower()
                ).split()
                if len(caption_text_list) < 3:
                    continue
                match = find_sublist_in_list(caption_text_list, transcript_text_list)
                if match != -1:
                    print(f"Found {caption_text_list} in transcript at index {match}")
                    transcript_index = match
                    youtube_caption_index = i
                    alignment_found = True
                    break

        # case 3: youtube caption length is much larger than the transcript. This means that the
        # video started before the transcript. So look through the beginning of the transcript
        # and try to find it in a caption
        else:
            for i in range(min(100, len(self._transcript_list))):
                transcript_text_list = [
                    self._remove_punctuation(word.lower())
                    for word in self._transcript_list[i : i + 3]
                ]
                for j, caption in enumerate(self._youtube_caption):
                    caption_text_list = self._remove_punctuation(
                        caption.text.lower()
                    ).split()
                    if len(caption_text_list) < 3:
                        continue
                    match = find_sublist_in_list(
                        transcript_text_list, caption_text_list
                    )
                    if match != -1:
                        print(
                            f"Found transcript {transcript_text_list} in youtube at {match}"
                        )
                        alignment_found = True
                        break
                if alignment_found:
                    transcript_index = match
                    youtube_caption_index = j
                    break

        if alignment_found:
            self._youtube_caption = self._youtube_caption[youtube_caption_index:]
            self._transcript_list = self._transcript_list[transcript_index:]

        return alignment_found

    def _align_captions(self) -> T.List[TranscriptChunk]:
        start_index = 0
        len_cur_text = 0
        aligned_transcripts: T.List[TranscriptChunk] = []

        for i in range(0, len(self._youtube_caption) - 1):

            cur_caption = self._youtube_caption[i]
            len_cur_text += len(cur_caption.text.split())
            # TODO(divyanshu): perhaps make this a setting to the aligner
            buffer = len_cur_text

            next_caption = self._youtube_caption[i + 1]
            if not (self._validate_youtube_caption(next_caption)):
                continue
            next_text = next_caption.text
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
                if len(aligned_transcripts) > 0:
                    start_time = aligned_transcripts[-1].end
                else:
                    start_time = self._youtube_caption[0].start
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
