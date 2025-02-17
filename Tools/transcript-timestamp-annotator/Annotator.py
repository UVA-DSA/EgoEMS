from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QSlider,
    QApplication, QInputDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
import json
import sys
import vlc

class VideoAnnotationTool(QMainWindow):
    def __init__(self, video_path, json_path, tasks_path):
        super().__init__()
        self.setWindowTitle("Video Annotation Tool")
        self.setGeometry(100, 100, 800, 600)

        # VLC Player Setup
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self.media = self.instance.media_new(video_path)
        self.media_player.set_media(self.media)

        # Video Widget Setup
        video_widget = QWidget(self)
        self.video_widget = video_widget  # Store reference for resizing later
        layout = QVBoxLayout()
        layout.addWidget(video_widget)

        # Set VLC's video output to the widget's window handle
        if sys.platform.startswith("linux"):  # Linux
            self.media_player.set_xwindow(self.video_widget.winId())
        elif sys.platform == "win32":  # Windows
            self.media_player.set_hwnd(self.video_widget.winId())
        elif sys.platform == "darwin":  # macOS
            self.media_player.set_nsobject(int(self.video_widget.winId()))

        # Connect VLC events to your methods
        self.media_player.event_manager().event_attach(
            vlc.EventType.MediaPlayerTimeChanged, self.on_position_changed
        )
        self.media_player.event_manager().event_attach(
            vlc.EventType.MediaPlayerLengthChanged, self.on_duration_changed
        )

        # Controls
        self.playButton = QPushButton("Play/Pause")
        self.playButton.clicked.connect(self.toggle_playback)

        self.label = QLabel(
            "Press space to play/pause. Use 'b' to go back 5s and 'f' to go forward 5s. "
            "Press 's' to set a start time, press 'e' to set an end time, and press 'c' to edit the current utterance. "
            "Press 'a' to create a new utterance. Use the up and down arrow keys to cycle through utterances."
        )
        self.label.setWordWrap(True)  # Enable word wrapping for instructions

        self.annotation_label = QLabel("")  # Annotation label
        self.annotation_label.setWordWrap(True)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 1000)  # Placeholder range; will be updated dynamically
        self.positionSlider.sliderMoved.connect(self.set_position)

        # Add controls to layout
        layout.addWidget(self.positionSlider)
        layout.addWidget(self.playButton)
        layout.addWidget(self.label)
        layout.addWidget(self.annotation_label)

        # Main widget setup
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Load JSON data and tasks file path
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.tasks_path = tasks_path

        with open(tasks_path, 'r') as f:
            self.tasks = json.load(f)

        self.current_index = 0
        if len(self.data) > 0:
            self.update_annotation_label()
    
    def on_position_changed(self, event):
        """Callback for VLC MediaPlayerTimeChanged event."""
        self.position_changed()

    def on_duration_changed(self, event):
        """Callback for VLC MediaPlayerLengthChanged event."""
        self.duration_changed()

    def toggle_playback(self):
        """Toggle play/pause for the VLC media player."""
        if self.media_player.is_playing():
            self.media_player.pause()
        else:
            self.media_player.play()

    def set_position(self, position):
        """Set the playback position based on slider movement."""
        duration = self.media_player.get_length()  # Get video duration in milliseconds
        if duration > 0:
            seek_time = int(position * duration / 1000)  # Scale slider value to milliseconds
            self.media_player.set_time(seek_time)

    def update_annotation_label(self):
        """Update the annotation label with the current utterance."""
        if 0 <= self.current_index < len(self.data):
            utterance = self.data[self.current_index]
            role = utterance.get("Role", "Unknown")
            text = utterance.get("Utterance", "")
            start_t = utterance.get("start_t", "N/A")
            end_t = utterance.get("end_t", "N/A")
            label_text = f"Role: {role}\nUtterance: {text}\nStart: {start_t}\nEnd: {end_t}"
            self.annotation_label.setText(label_text)
    def delete_current_utterance(self):
        """Delete the current utterance and update the annotations."""
        if 0 <= self.current_index < len(self.data):
            # Remove the current utterance
            deleted_utterance = self.data.pop(self.current_index)
            print(f"Deleted utterance: {deleted_utterance}")

            # Adjust the index to point to a valid utterance
            if self.current_index >= len(self.data):
                self.current_index = max(0, len(self.data) - 1)  # Move to the last utterance if index is out of bounds

            # Update annotation label or show a message if no utterances remain
            if len(self.data) > 0:
                self.update_annotation_label()
            else:
                print("No more utterances.")
                self.annotation_label.setText("No more utterances.")
        else:
            print("No utterance to delete.")

    def edit_current_utterance(self):
        """Edit the current utterance's Role and Utterance fields."""
        if 0 <= self.current_index < len(self.data):
            current_utterance = self.data[self.current_index]

            # Edit Role
            new_role, ok_role = QInputDialog.getText(
                self, "Edit Role", "Enter new role:", text=current_utterance.get("Role", "")
            )
            if ok_role and new_role.strip():
                current_utterance["Role"] = new_role.strip()

            # Edit Utterance
            new_utterance, ok_utterance = QInputDialog.getMultiLineText(
                self, "Edit Utterance", "Enter new utterance:", text=current_utterance.get("Utterance", "")
            )
            if ok_utterance and new_utterance.strip():
                current_utterance["Utterance"] = new_utterance.strip()

            print(f"Updated annotation: {current_utterance}")
            self.update_annotation_label()

    def save_json_and_quit(self):
        """Save the updated JSON data and quit the application."""
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=4)
        sys.exit(0)
        
    def is_task_complete(self):
        """Check if all utterances have valid start and end times."""
        for utterance in self.data:
            if not utterance.get('start_t') or not utterance.get('end_t'):
                return False
        return True

    def save_and_switch_task(self):
        """Save current task and switch to the next one."""
        
        # Save current annotations
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=4)
        
        print(f"Annotations saved for {self.json_path}")

        # Remove task from tasks.json if complete
        if self.is_task_complete():
            print(f"Task completed: {self.json_path}")
            for task in self.tasks:
                if task['annotation'] == self.json_path:
                    self.tasks.remove(task)
                    break
            
            with open(self.tasks_path, 'w') as f:
                json.dump(self.tasks, f, indent=4)
        
            print(f"Task removed from {self.tasks_path}")

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Space:
            self.toggle_playback()

        elif event.key() == Qt.Key_B:  # Go back 5 seconds
            current_time = self.media_player.get_time()
            self.media_player.set_time(max(0, current_time - 5000))

        elif event.key() == Qt.Key_F:  # Go forward 5 seconds
            current_time = self.media_player.get_time()
            duration = self.media_player.get_length()
            self.media_player.set_time(min(duration, current_time + 5000))

        elif event.key() == Qt.Key_S:  # Set start time
            if 0 <= self.current_index < len(self.data):
                start_time = self.format_time(self.media_player.get_time())
                self.data[self.current_index]['start_t'] = start_time
                print(f"Start time set to {start_time}")
                self.update_annotation_label()

        elif event.key() == Qt.Key_E:  # Set end time
            if 0 <= self.current_index < len(self.data):
                end_time = self.format_time(self.media_player.get_time())
                self.data[self.current_index]['end_t'] = end_time
                print(f"End time set to {end_time}")
                if (self.current_index + 1) < len(self.data): 
                    # Move to next utterance automatically after setting end time.
                    self.current_index += 1  
                    self.update_annotation_label()
                else:
                    print("No more utterances.")
                    self.annotation_label.setText("No more utterances.")

        elif event.key() == Qt.Key_C:  # Change current utterance
            self.edit_current_utterance()

        elif event.key() == Qt.Key_A:  # Add a new utterance
            new_utterance = {
                "Role": "New Role",
                "Utterance": "New Utterance",
                "start_t": "",
                "end_t": ""
            }

            self.data.insert(self.current_index + 1, new_utterance)
            print("New utterance added.")
            self.current_index += 1
            print(f"Moved to next utterance (index: {self.current_index})")
            self.update_annotation_label()

        elif event.key() == Qt.Key_D:
            self.delete_current_utterance()

        elif event.key() == Qt.Key_Up:  # Go to the previous utterance
            if self.current_index > 0:
                self.current_index -= 1
                print(f"Moved to previous utterance (index: {self.current_index})")
                self.update_annotation_label()
            else:
                print("Already at the first utterance.")

        elif event.key() == Qt.Key_Down:  # Go to the next utterance
            if self.current_index < len(self.data) - 1:
                self.current_index += 1
                print(f"Moved to next utterance (index: {self.current_index})")
                self.update_annotation_label()
            else:
                print("Already at the last utterance.")

        elif event.key() == Qt.Key_Left:  # Go backward 0.5 seconds
            current_time = self.media_player.get_time()
            self.media_player.set_time(max(0, current_time - 500))

        elif event.key() == Qt.Key_Right:  # Go forward 0.5 seconds
            current_time = self.media_player.get_time()
            duration = self.media_player.get_length()
            self.media_player.set_time(min(duration, current_time + 500))

        elif event.key() == Qt.Key_Q:  # Quit and save
            self.save_json_and_quit()

        elif event.key() == Qt.Key_N:  # Save and switch task
            print("Saving and switching task...")
            self.save_and_switch_task()

            # Check if there are remaining tasks in tasks.json
            if len(self.tasks) > 0:
                next_task = self.tasks[0]
                video_path = next_task['video']
                json_path = next_task['annotation']

                # Reload the application with the new task
                print(f"Switching to next task: {json_path}")
                self.close()
                
                # Open a new instance of VideoAnnotationTool for the next task
                new_window = VideoAnnotationTool(video_path, json_path, self.tasks_path)
                new_window.show()
                
            else:
                print("No more tasks available.")

    def format_time(self, milliseconds):
        """Format time in milliseconds to HH:MM:SS format."""
        seconds = milliseconds // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def position_changed(self):
        """Update slider position as video plays."""
        duration = self.media_player.get_length()
        if duration > 0:
            current_position = int(self.media_player.get_time() * 1000 / duration)
            self.positionSlider.setValue(current_position)

    def duration_changed(self):
        """Set slider range based on video duration."""
        duration = self.media_player.get_length()
        if duration > 0:
            self.positionSlider.setRange(0, 1000)  # Slider range normalized to [0, 1000]

# Helper function to add missing timestamps to utterances
def add_timestamps(data):
    """Ensure all utterances have 'start_t' and 'end_t' fields."""
    for utterance in data:
        if 'start_t' not in utterance:
            utterance['start_t'] = ''  # Initialize if not present
        if 'end_t' not in utterance:
            utterance['end_t'] = ''  # Initialize if not present
    return data

# Function to run the main loop for processing tasks
def main(tasks_path):
    """Process tasks from tasks.json and handle each task."""
    # Load tasks from the JSON file
    with open(tasks_path, 'r') as f:
        tasks = json.load(f)

    # Iterate through each task in the list
    for task in tasks:
        video_path = task['video']
        json_path = task['annotation']

        # Load and ensure timestamps are added to the annotation file
        with open(json_path, 'r') as annotation_file:
            data = json.load(annotation_file)
            data = add_timestamps(data)  # Ensure timestamps are present

        # Save back the updated annotations with timestamps
        with open(json_path, 'w') as annotation_file:
            json.dump(data, annotation_file, indent=4)

        # Initialize and process the VideoAnnotationTool for this task
        app = QApplication(sys.argv)
        window = VideoAnnotationTool(video_path, json_path, tasks_path)
        window.show()
        
        sys.exit(app.exec_())

# Main entry point for the application
if __name__ == "__main__":
    tasks_file = "tasks.json"  # Path to your tasks.json file
    main(tasks_file)

        
