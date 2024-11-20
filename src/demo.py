import subprocess
import logging
from contextlib import contextmanager
import sys
import threading
import queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whisper_process.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


@contextmanager
def whisper_process():
    cmd = ["../modules/whisper.cpp/stream", "-m", "../modules/whisper.cpp/models/ggml-base.en.bin"]
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # This ensures text output instead of bytes
            bufsize=1  # Line buffered
        )

        # Create queues for stdout and stderr
        out_queue = queue.Queue()
        err_queue = queue.Queue()

        # Define stream reader thread
        def stream_reader(input_stream, output_queue, stream_name):
            try:
                for line_text in input_stream:
                    output_queue.put(line_text.strip())
                    logging.info(f"{stream_name}: {line_text.strip()}")
            except Exception as e:
                logging.error(f"Error reading {stream_name}: {e}")
            finally:
                input_stream.close()

        # Start reader threads
        out_thread = threading.Thread(
            target=stream_reader,
            args=(proc.stdout, out_queue, "STDOUT")
        )
        err_thread = threading.Thread(
            target=stream_reader,
            args=(proc.stderr, err_queue, "STDERR")
        )

        out_thread.daemon = True
        err_thread.daemon = True
        out_thread.start()
        err_thread.start()

        yield proc, out_queue, err_queue

    except Exception as e:
        logging.error(f"Failed to start whisper process: {e}")
        raise
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            out_thread.join(timeout=1)
            err_thread.join(timeout=1)

            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()


if __name__ == "__main__":
    with whisper_process() as (process, stdout_queue, stderr_queue):
        # Process is now running and logs are being captured
        try:
            while True:
                # Check stdout
                try:
                    stdout_line = stdout_queue.get_nowait()
                    print(f"Got output: {stdout_line}")
                except queue.Empty:
                    pass

                # Check stderr
                try:
                    stderr_line = stderr_queue.get_nowait()
                    print(f"Got error: {stderr_line}")
                except queue.Empty:
                    pass

                # Check if process is still alive
                if process.poll() is not None:
                    break

        except KeyboardInterrupt:
            logging.info("Process interrupted by user")
