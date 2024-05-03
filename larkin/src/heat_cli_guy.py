from pathlib import Path

from colorama import Fore, Style
import os
import re
import logging
import click


class InteractiveCLI:
    INTRO_MESSAGE = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    █  Welcome to the Milky Electric Path Parser!  █
    █ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ █
    █ - Follow the prompts to navigate Parser      █
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    BORDERS = (
        "█" * 48
        + "█" * 2
        + "█" * 48
        + "█" * 2
        + "█" * 48
        + "█" * 2
        + "█" * 48
        + "█" * 2
    )

    def __init__(self):
        self.start_path = ""
        self.end_path = ""
        self.setup_logger()

    def setup_logger(self, debug=False):
        logger = logging.getLogger(__name__)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler("debug.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    def display_start_message(self):
        """
        Display the start message of the Milky Electric Path Parser.

        This method prints the introduction message of the parser, including the welcome message and instructions for navigating the parser.

        Returns:
            None
        """
        MAGENTA_COLOR = Fore.MAGENTA
        YELLOW_COLOR = Fore.YELLOW
        RESET_COLOR = Style.RESET_ALL

        click.echo(f"{MAGENTA_COLOR}{InteractiveCLI.BORDERS}")
        click.echo(f"{YELLOW_COLOR}{InteractiveCLI.INTRO_MESSAGE}")
        click.echo(f"{MAGENTA_COLOR}{InteractiveCLI.BORDERS}{RESET_COLOR}")

    def get_directory_input(self):
        while True:
            try:
                self.start_path = click.prompt(
                    "Enter the full path to the start directory",
                    type=click.Path(
                        exists=True,
                        file_okay=False,
                        dir_okay=True,
                        readable=True,
                        writable=True,
                    ),
                )
                self.end_path = click.prompt(
                    "Enter the full path to the end directory",
                    type=click.Path(
                        exists=True,
                        file_okay=False,
                        dir_okay=True,
                        readable=True,
                        writable=True,
                    ),
                )
                if self.validate_directories():
                    click.echo(
                        click.style(
                            "Valid directories provided. Operation successful.",
                            fg="green",
                        )
                    )
                    break
                click.echo(
                    click.style(
                        "Invalid directories provided. Please check and try again.",
                        fg="red",
                    )
                )
            except (EOFError, KeyboardInterrupt):
                click.echo(
                    click.style("Input interrupted. Please try again.", fg="red")
                )

    def validate_directories(self):
        """
        Validates the start and end directories.

        Raises:
            Exception: If the directories are not valid.
        """
        if not os.path.isdir(self.start_path) or not os.path.isdir(self.end_path):
            raise Exception("Invalid directories provided.")

    def parse_python_code(self, filepath):
        try:
            sections = Path(filepath).read_text(encoding="utf-8").split("New chat")
        except Exception as e:
            logging.exception("Error reading file")
            return

        for idx, section in enumerate(sections):
            try:
                if python_code := self.extract_python_from_text(section):
                    self.save_to_file(python_code, idx)
            except Exception as e:
                logging.exception("Error saving file")

        return True

    def process_directories(self):
        with open(self.start_path, "r", encoding="utf-8") as file:
            for idx, section in enumerate(file.read().split("New chat")):
                try:
                    if python_code := self.extract_python_from_text(section):
                        self.save_to_file(python_code, idx)
                except Exception as e:
                    logging.exception("Error saving file")

        return True

    def extract_python_from_text(self, text):
        return "\n".join(re.findall(r"```python\s+(.*?)\s+```", text, re.DOTALL))

    def save_to_file(self, code, index):
        output_filename = f"Chat{index + 1}_ExtractedPython.py"
        output_path = os.path.join(self.end_path, output_filename)
        if not os.path.exists(output_path):
            try:
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(code)
            except Exception as e:
                logging.exception("Error saving file")
        else:
            # Handle file already exists case
            ...


def main():
    cli = InteractiveCLI()
    cli.display_start_message()
    try:
        cli.get_directory_input()
    except Exception:
        logging.exception("Error getting directory input")
    cli.process_directories()
    return True


if __name__ == "__main__":
    main()
