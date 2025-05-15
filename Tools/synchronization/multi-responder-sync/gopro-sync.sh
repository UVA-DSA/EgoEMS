ffmpeg -f concat -safe 0 -i inputs.txt -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k output.mp4
