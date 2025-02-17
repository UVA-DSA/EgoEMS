# Video Annotation Tool

This project is a video annotation tool that allows users to annotate a transcript for a video with timestamps.. The tool uses VLC for video playback and PyQt5 for the graphical user interface.

## Downloading Files from a Remote Server
Create a file called `files.txt` that contains a new-line separated list of file paths. Modify any logic in `fetch_data.py` in order to match your directory specifications.

## Features

- Play/pause video playback
- Set start and end times for utterances
- Navigate through utterances
- Save annotations to JSON files
- Switch between tasks

## Requirements

- Python 3.x
- PyQt5
- python-vlc

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/asatpathy314/transcript-timestamping-tool.git
    cd video-annotation-tool
    ```

2. Install the required packages:

## Usage

1. Prepare a `tasks.json` file with the following structure:
    ```json
    [
        {
            "video": "path/to/video_1.mp4",
            "annotation": "path/to/annotation_1.json"
        },
        ...
        {
            "video": "path/to/video_n.mp4",
            "annotation": "path/to/annotation_n.json"
        },
    ]
    ```

2. Run the tool:
    ```sh
    python Annotator.py
    ```

## Keyboard Shortcuts

- `Space`: Toggle play/pause
- `b`: Go back 5 seconds
- `f`: Go forward 5 seconds
- `s`: Set start time
- `e`: Set end time and go to next utterance.
- `c`: Edit current utterance
- `a`: Add a new utterance
- `Up`: Go to the previous utterance
- `Down`: Go to the next utterance
- `Left`: Go backward 0.5 seconds
- `Right`: Go forward 0.5 seconds
- `q`: Quit and save
- `n`: Save and switch task

## Project Structure

- `Annotator.py`: Main application file
- `example_tasks.json`: Example tasks file
- `AnnotationFiles/`: Directory for annotation JSON files
- `VideoFiles/`: Directory for video files

## License

This project is licensed under the MIT License.
