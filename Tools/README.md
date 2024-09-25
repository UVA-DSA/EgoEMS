### FFMPEG extract RGB stream from Kinect Recording
``` ffmpeg -i 2024-09-05-19-07-43_trimmed.mkv -map 0:v:0 -c:v libx264 -crf 23 -preset fast rgb_stream.mp4 ```

### FFMPEG combine GoPro and Kinect RGB view 
```bash
ffmpeg -i GoPro/GX010321_trimmed.mp4 -i Kinect/rgb_stream.mp4 \
-filter_complex "[0:v]scale=1920:1080,fps=30,setpts=PTS-STARTPTS[v0]; \
[1:v]scale=1920:1080,fps=30,setpts=PTS-STARTPTS[v1]; \
[v0][v1]hstack=inputs=2,format=yuv420p[v]" \
-map "[v]" -map 0:a? -shortest output.mp4

```