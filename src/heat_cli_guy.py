from colorama import Fore, Style
import os
import re
import logging

class InteractiveCLI:
    INTRO_MESSAGE = """
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    █  Welcome to the Milky Electric Path Parser!  █
    █ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ █
    █ - Follow the prompts to navigate Parser      █
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    BORDERS = "█" * 48 + "█" * 2 + "█" * 48 + "█" * 2 + "█" * 48 + "█" * 2 + "█" * 48 + "█" * 2

    def __init__(self):
        self.start_path = ''
        self.end_path = ''
        self.setup_logger()

    @staticmethod
    def setup_logger(logger=logging.getLogger(__name__), debug=False):
        if debug:
            logger.setLevel(logging.DEBUG)
        logger.setLevel(logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler("debug.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    def display_start_message(self, click=None):
        click.echo(Fore.MAGENTA + InteractiveCLI.BORDERS)
        click.echo(Fore.YELLOW + InteractiveCLI.INTRO_MESSAGE)
        click.echo(Fore.MAGENTA + InteractiveCLI.BORDERS + Style.RESET_ALL)

    def get_directory_input(self):
        while True:
            self.start_path = click.prompt("Enter the full path to the start directory")
            self.end_path = click.prompt("Enter the full path to the end directory")
            if self.validate_directories():
                break
            click.echo(click.style("Invalid directories provided. Please check and try again.", fg='red'))

    def validate_directories(self):
        return Path(self.start_path).is_dir() and Path(self.end_path).is_dir() and Path(
            self.end_path).resolve().parent == Path(self.start_path).resolve()


class PythonCodeExtractor:
    def __init__(self, cli):
        self.cli = cli

    def process_directories(self):
        for filepath in Path(self.cli.start_path).rglob('*.txt'):
            self.parse_python_code(filepath)

    def parse_python_code(self, filepath):
        try:
            sections = Path(filepath).read_text(encoding='utf-8').split("New chat")
        except Exception as e:
            logging.exception("Error reading file")
            return

        for idx, section in enumerate(sections):
            try:
                python_code = self.extract_python_from_text(section)
                if python_code:
                    self.save_to_file(python_code, idx)
            except Exception as e:
                logging.exception("Error saving file")

        return True

    def extract_python_from_text(self, text):
        return '\n'.join(re.findall(r'```python\s+(.*?)\s+```', text, re.DOTALL))

    def save_to_file(self, code, index):
        output_filename = "Chat{index + 1}_ExtractedPython.py"
        output_path = os.path.join(self.cli.end_path, output_filename)
        Path(output_path).write_text(code, encoding='utf-8')


def main():
    cli = InteractiveCLI()
    cli.display_start_message(click)
    cli.get_directory_input()
    extractor = PythonCodeExtractor(cli)
    extractor.process_directories()


if __name__ == "__main__":
    main()