from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
import time
import random

console = Console()

class AssistantStatus:
    def __init__(self):
        self.current_action = "IDLE"
        self.current_text = ""
        self.start_time = time.time()
        self.duration_limits = {
            "LISTENING": 10,  # 10 seconds for listening
            "ANALYZING": 2,   # 2 seconds for analyzing
            "SPEAKING": 15,   # 15 seconds for speaking
            "IDLE": 1        # 1 second for idle
        }
        
    def update(self, action, text=""):
        self.current_action = action
        self.current_text = text
        self.start_time = time.time()
        
    def should_transition(self):
        current_duration = time.time() - self.start_time
        return current_duration >= self.duration_limits.get(self.current_action, 0)

def create_main_display(status):
    # Calculate duration of current action
    duration = time.time() - status.start_time
    time_left = status.duration_limits.get(status.current_action, 0) - duration
    
    # Define styling for different states
    styles = {
        "LISTENING": {
            "icon": "🎧",
            "color": "cyan",
            "animation": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.time() * 4) % 10],
        },
        "ANALYZING": {
            "icon": "🔄",
            "color": "yellow",
            "animation": "⠋⠙⠚⠞⠖⠦⠴⠲⠳⠓"[int(time.time() * 4) % 10],
        },
        "SPEAKING": {
            "icon": "🗣️",
            "color": "green",
            "animation": "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"[int(time.time() * 4) % 15],
        },
        "IDLE": {
            "icon": "💭",
            "color": "grey70",
            "animation": "●",
        }
    }
    
    style = styles.get(status.current_action, styles["IDLE"])
    
    # Create the main status display
    status_text = Text()
    status_text.append("\n" * 2)
    status_text.append(f"{style['icon']} Current Status: ", style="bold white")
    status_text.append(f"{status.current_action}\n", style=f"bold {style['color']}")
    status_text.append("\n")
    status_text.append(f"{style['animation']} ", style=style['color'])
    status_text.append(status.current_text, style=style['color'])
    status_text.append("\n" * 2)
    status_text.append("Duration: ", style="bold white")
    status_text.append(f"{duration:.1f}s", style="bold blue")
    status_text.append("  |  Time Left: ", style="bold white")
    status_text.append(f"{time_left:.1f}s", style="bold red")
    status_text.append("\n" * 2)
    
    return Panel(
        status_text,
        title=f"AI Assistant Monitor",
        border_style=style['color'],
        padding=(1, 2),
    )

def run_assistant_monitor():
    status = AssistantStatus()
    
    with Live(refresh_per_second=10, screen=True) as live:
        while True:
            try:
                # Check if it's time to transition to next state
                if status.should_transition():
                    if status.current_action == "IDLE":
                        status.update("LISTENING", "Listening to your voice input...")
                    elif status.current_action == "LISTENING":
                        status.update("ANALYZING", "Processing and analyzing your request...")
                    elif status.current_action == "ANALYZING":
                        status.update("SPEAKING", "Here's what I found based on your input...")
                    elif status.current_action == "SPEAKING":
                        status.update("IDLE", "Ready for next command")
                
                # Update the display
                layout = Layout()
                layout.update(create_main_display(status))
                live.update(layout)
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                console.print("\n[bold red]Shutting down assistant...[/bold red]")
                break

if __name__ == "__main__":
    console.print("[bold green]Starting AI Assistant Monitor...[/bold green]")
    time.sleep(1)
    run_assistant_monitor() 