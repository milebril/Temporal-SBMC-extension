import ffmpeg

input_dir = '/home/emil/Documents/Remote-SBMC/output/emil/training_peters_theirs/*.png'
output_dir = '/home/emil/Desktop/output_peters.mp4'

(
    ffmpeg
    .input(input_dir, pattern_type='glob', framerate=15.0)
    .output(output_dir) # , **{'b:v': 4000}
    .run()
)