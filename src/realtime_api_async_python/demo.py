from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box
from rich.layout import Layout
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from datetime import datetime
import random
import time

console = Console()

def get_server_status():
    statuses = [
        ("🟢 Online", "green"),
        ("🔴 Offline", "red"),
        ("🟡 Maintenance", "yellow"),
    ]
    return random.choice(statuses)

def generate_table():
    table = Table(show_header=True, header_style="bold blue", box=box.DOUBLE_EDGE)
    table.add_column("ID", style="dim")
    table.add_column("Server", style="green")
    table.add_column("Status", style="red")
    table.add_column("Load", style="cyan")
    table.add_column("Response Time", style="magenta")
    
    for i in range(1, 4):
        status, color = get_server_status()
        load = f"{random.uniform(0, 100):.1f}%"
        response = f"{random.uniform(10, 500):.0f}ms"
        table.add_row(
            str(i),
            f"Server {i}",
            status,
            load,
            response
        )
    return table

def generate_metrics_panel():
    metrics = [
        f"Active Users: {random.randint(100, 1000)}",
        f"CPU Usage: {random.uniform(0, 100):.1f}%",
        f"Memory: {random.uniform(0, 100):.1f}%",
        f"Network I/O: {random.uniform(0, 1000):.1f} MB/s"
    ]
    return Panel("\n".join(metrics), title="System Metrics")

def live_monitor():
    with Live(refresh_per_second=1) as live:
        while True:
            # Create layout
            layout = Layout()
            layout.split_column(
                Layout(name="header"),
                Layout(name="main"),
                Layout(name="footer")
            )

            # Header
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = Panel(
                f"[bold magenta]System Monitor[/bold magenta]\n"
                f"[blue]Last Updated: {current_time}[/blue]",
                style="bold"
            )
            layout["header"].update(header)

            # Main content
            main_layout = Layout()
            main_layout.split_row(
                Layout(generate_table()),
                Layout(generate_metrics_panel())
            )
            layout["main"].update(main_layout)

            # Footer with status messages
            status_messages = [
                "[green]✓ Database connection stable[/green]",
                f"[yellow]⚡ Processing requests: {random.randint(10, 100)}[/yellow]",
                "[blue]☁ Cloud sync active[/blue]"
            ]
            footer = Panel("\n".join(status_messages), title="System Status")
            layout["footer"].update(footer)

            # Update the live display
            live.update(layout)
            time.sleep(1)

if __name__ == "__main__":
    try:
        console.print("[bold green]Starting System Monitor...[/bold green]")
        time.sleep(1)
        live_monitor()
    except KeyboardInterrupt:
        console.print("\n[bold red]Shutting down monitor...[/bold red]")