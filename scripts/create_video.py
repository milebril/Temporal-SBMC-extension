import ffmpeg

input_dir = '/home/emil/Documents/Remote-SBMC/output/emil/training_peters_final/*.png'
output_dir = '/home/emil/Desktop/output_peters_final.mp4'

(
    ffmpeg
    .input(input_dir, pattern_type='glob', framerate=30.0)
    .output(output_dir) # , **{'b:v': 4000}
    .run()
)


# join-ffmpeg() {
#   ffmpeg \
#     -framerate 1 \
#     -pattern_type glob \
#     -i '_out/4d/remote-bathroom-plyrotate/*.800x800.png' \
#     -c:v libx264 \
#     -r 30 \
#     -pix_fmt yuv420p \
#     _out/4d/remote-bathroom-plyrotate.mp4
# }

