import mimetypes


def load_audio_as_bytes(file_path: str) -> tuple[bytes, str]:
    """
    Load an audio file of any format and return its raw bytes.

    Args:
    - file_path (str): Path to the audio file.

    Returns:
    - bytes: Raw audio data in bytes format.
    - str: MIME type of the audio file.
    """
    # Guess MIME type from file extension
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        raise ValueError("Could not determine MIME type for the file.")

    # Read the file as raw bytes
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes, mime_type
