A program that keeps running and detects audio and take various actions on it.
USE silero_vad for it and load it from the data repo

There are global state like 
1. listening
2. idle
3. thinking
4. output
5. error

Use the default mic and speaker available on the system for the audio.
If output is happening the program can not listen to any more audio.

Once the program is done listening it goes into thinking state
During that state it make some api calls and get the output.



