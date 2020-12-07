import ffmpeg

(
    ffmpeg
    .input('/home/emil/Documents/sbmc/output/emil/dataviz_sequence_duidelijk/denoised/*.png', pattern_type='glob', framerate=24)
    .output('/home/emil/Documents/sbmc/output/emil/dataviz_sequence_duidelijk/denoised/movie.mkv') # , **{'b:v': 4000}
    .run()
)