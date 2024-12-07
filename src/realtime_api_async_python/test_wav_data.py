def get_wav_metadata(wav_file_path):
    """
    Get metadata information from a WAV file.
    
    Args:
        wav_file_path (str): Path to the WAV file
        
    Returns:
        dict: Dictionary containing WAV file metadata
    """
    import wave
    
    try:
        with wave.open(wav_file_path, 'rb') as wav_file:
            metadata = {
                'channels': wav_file.getnchannels(),
                'sample_width': wav_file.getsampwidth(),
                'frame_rate': wav_file.getframerate(),
                'n_frames': wav_file.getnframes(),
                'compression_type': wav_file.getcomptype(),
                'compression_name': wav_file.getcompname()
            }
            
            # Calculate duration in seconds
            metadata['duration'] = metadata['n_frames'] / metadata['frame_rate']
            
            return metadata
            
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {wav_file_path}")
        return None

# Example usage:
files_path = ["./recordings/1.wav", "./recordings/2.wav", "./recordings/3.wav"]

for file_path in files_path:
    metadata = get_wav_metadata(file_path)
    if metadata:
        print(f"\nMetadata for {file_path}:")
        print("Channels:", metadata['channels'])
        print("Sample Width:", metadata['sample_width'], "bytes")
        print("Frame Rate:", metadata['frame_rate'], "Hz")
        print("Number of Frames:", metadata['n_frames'])
        print("Duration:", round(metadata['duration'], 2), "seconds")
        print("Compression Type:", metadata['compression_type'])
        print("Compression Name:", metadata['compression_name'])
    else:
        print(f"\nFailed to get metadata for {file_path}")
