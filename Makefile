# This Makefile defines rules to build and run a Docker environment for easier
# reproducibility, as well as commands demonstrating how to use the code
# (commands starting with `demo/`).

OUTPUT?=$(shell pwd)/output
DATA?=$(shell pwd)/data
PBRT?=$(shell pwd)/pbrt/bin/pbrt
OBJ2PBRT?=$(shell pwd)/pbrt/bin/obj2pbrt

# Data server
REMOTE=https://data.csail.mit.edu/graphics/sbmc

# Path to comparison code
SEN2011?=_extras/comparisons/methods/2011_sen_rpf
ROUSSELLE2012?=_extras/comparisons/methods/2012_rousselle_nlm
KALANTARI2015?=_extras/comparisons/methods/2015_kalantari_lbf
BITTERLI2016?=_extras/comparisons/methods/nfor_fromdocker

# Checks whether docker version supports the --gpus option
check_docker_version:
	./scripts/check_docker_version.sh

# Install the required extension for CUDA on Docker
nvidia_docker: check_docker_version
	./scripts/install_nvidia_docker.sh

# To facilitate environment setup, build and use this dockerfile
# !! Requires NVIDIA's docker extension !!
docker_build:
	@docker build -f dockerfiles/cuda-sbmc.dockerfile -t sbmc_cuda_new .

# To facilitate environment setup, build and use this dockerfile
# !! Requires NVIDIA's docker extension !!
docker_build_cpu:
	@docker build -f dockerfiles/cpu-sbmc.dockerfile -t sbmc_cpu .

$(OUTPUT):
	mkdir -p $(OUTPUT)

$(DATA):
	mkdir -p $(DATA)

# This target launches a fully configured docker instance,
# mounts $(OUTPUT) as read-write volume and $(DATA) as readonly for persisten I/O.
# Once logged into the docker instance, you can run any of the `make demo/*`
# commands.
docker_run:  docker_build $(OUTPUT) $(DATA)
	@docker run --gpus all --name sbmc_cuda_app --rm \
		-v $(OUTPUT):/sbmc_app/output \
		-v $(DATA):/sbmc_app/data \
		--ipc=host \
		-p 2001:2001 \
		-it sbmc_cuda_new

docker_run_cpu:  docker_build_cpu $(OUTPUT) $(DATA)
	@docker run --name sbmc_cpu_app --rm \
		-v $(OUTPUT):/sbmc_app/output \
		-v $(DATA):/sbmc_app/data \
		--ipc=host \
		-p 2002:2001 \
		-it sbmc_cpu

clean:
	rm -rf dist .pytest_cache sbmc.egg-info build sbmc/halide_ops.*.so

test:
	pytest tests

.PHONY: demo/render_bins demo/render_reference \
	demo/visualize demo/denoiser demo/train demo/train_kpcn \
	demo/render_samples server clean nvidia_docker demo/data

# -----------------------------------------------------------------------------

demo/render_samples_emil:
	@python scripts/render_samples.py $(PBRT) \
		$(DATA)/demo/scenes/mblur.pbrt \
		$(OUTPUT)/demo/emil_samples \
		--tmp_dir $(OUTPUT)/tmp --height 512 --width 512 --spp 4 \
		--gt_spp 4 --verbose

demo/render_exr_emil:
	@python scripts/render_exr.py $(PBRT) \
		$(DATA)/demo/scenes/mblur.pbrt \
		$(OUTPUT)/emil/out.exr \
		--tmp_dir $(OUTPUT)/tmp --height 512 --width 512 --spp 512

render_png:
	@python scripts/render_samples.py $(PBRT) \
		$(DATA)/demo/scenes/mblur.pbrt \
		$(OUTPUT)/emil/samples/render \
		--tmp_dir $(OUTPUT)/tmp --height 512 --width 512 --spp 8 \
		--gt_spp 1 --verbose
	@cd $(OUTPUT)/emil/samples && find . -name "*.bin" > filelist.txt
	@python scripts/visualize_dataset.py \
		${OUTPUT}/emil/samples/ \
		${OUTPUT}/emil/renders --spp 512

generate_and_render:
	@rm -rf $(OUTPUT)/emil/training_scenes
	@python scripts/generate_training_data.py \
		$(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets \
		$(OUTPUT)/emil/training_scenes \
		--count 10 --spp 4 --gt_spp 4 --height 128 --width 128
	@cd $(OUTPUT)/emil/training_scenes && find . -name "*.bin" > filelist.txt
	@python scripts/visualize_dataset.py \
		$(OUTPUT)/emil/training_scenes \
		$(OUTPUT)/emil/dataviz --spp 4

render_training_set:
	@python scripts/visualize_dataset.py \
		$(OUTPUT)/emil/training_scenes \
		$(OUTPUT)/emil/dataviz --spp 4

generate_render_denoise:
	@rm -rf $(OUTPUT)/emil/training_scenes
	@python scripts/generate_training_data.py \
		$(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets \
		$(OUTPUT)/emil/training_scenes \
		--count 1 --spp 4 --gt_spp 512 --height 512 --width 512 --verbose --custom --no-clean
	@cd $(OUTPUT)/emil/training_scenes && find . -name "*.bin" > filelist.txt
	@python scripts/visualize_dataset.py \
		$(OUTPUT)/emil/training_scenes \
		$(OUTPUT)/emil/compare/ --spp 4
	@python scripts/denoise_sequence.py \
		--input $(OUTPUT)/emil/training_scenes \
		--output $(OUTPUT)/emil/compare/ \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/gharbi2019_sbmc
	@python scripts/denoise_sequence.py \
		--input $(OUTPUT)/emil/training_scenes \
		--output $(OUTPUT)/emil/compare/bako \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/bako2017_finetuned

generate_render:
	@rm -rf $(OUTPUT)/emil/training_sequence_tmp
	@python scripts/generate_training_sequence.py \
		$(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets \
		$(OUTPUT)/emil/training_sequence_tmp \
		--count 5 --frames 1 --spp 4 --gt_spp 4 --width 128 --height 128 --no-clean
	@cd $(OUTPUT)/emil/training_sequence_tmp && find . -name "*.bin" | sort -V > filelist.txt
	@python scripts/visualize_dataset.py \
		$(OUTPUT)/emil/training_sequence_tmp/render_samples_seq \
		$(OUTPUT)/emil/dataviz_sequence --spp 4 --frames 30


generate_training_sequence:
	@rm -rf $(OUTPUT)/emil/training_sequence_final
	@python scripts/generate_training_sequence.py \
		$(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets \
		$(OUTPUT)/emil/training_sequence_final \
		--count 100 --frames 5 --spp 4 --gt_spp 1024 --width 128 --height 128 --no-clean
	@cd $(OUTPUT)/emil/training_sequence_final && find . -name "*.bin" | sort -V > filelist.txt

generate_validation_sequence:
	@rm -rf $(OUTPUT)/emil/validation_sequence_wall
	@python scripts/generate_training_sequence.py \
		$(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets \
		$(OUTPUT)/emil/validation_sequence_wall \
		--count 1 --frames 10 --spp 4 --gt_spp 512 --width 256 --height 256 --no-clean
	@cd $(OUTPUT)/emil/validation_sequence_wall && find . -name "*.bin" | sort -V > filelist.txt

show_all: visualize_sequence denoise_sequence_pretrained denoise_sequence_peters 
	@python scripts/make_compare_video.py 

# Visualizes the given dataset specified at the location in the first argument
# Ouputs the ground thruth render as well as the low spp render
visualize_sequence:
	@python scripts/visualize_dataset.py \
		$(OUTPUT)/emil/training_sequence_tmp/render_samples_seq \
		$(OUTPUT)/emil/dataviz_sequence --spp 4 --frames 30
		# $(OUTPUT)/emil/validation_sequence/render_samples_seq \
		# $(OUTPUT)/emil/dataviz_val_sequence --spp 4

# Denoises a given sequence using the pretrained model from Gharbi et al
denoise_sequence_pretrained:
	@python scripts/denoise.py \
		--input $(OUTPUT)/emil/training_sequence_very_mat/render_samples_seq \
		--output $(OUTPUT)/emil/dataviz_sequence/denoised/pretrained/ \
		--spp 4 --sequence \
		--checkpoint $(DATA)/pretrained_models/gharbi2019_sbmc \
		--frames 10

# Denoises the given sequence using the temporal extension to SBMC
denoise_sequence_peters:
	@python scripts/denoise.py \
		--input $(OUTPUT)/emil/training_sequence_outdoor/render_samples_seq \
		--output $(OUTPUT)/emil/dataviz_sequence/denoised/peters/ \
		--spp 4 --sequence --temporal \
		--checkpoint $(OUTPUT)/emil/trained_models/peters_all_loaded.pth \
		--frames 5

# Trains the recurrent SBMC model
train_emil:
	@python scripts/train.py \
		--checkpoint_dir $(OUTPUT)/emil/training_peters_cornell \
		--data $(OUTPUT)/emil/training_sequence_cornell/filelist.txt \
		--val_data $(OUTPUT)/emil/validation_sequence/filelist.txt \
		--env sbmc_ours --port 2001 --bs 1 --constant_spp --emil_mode \
		--spp 4

# Trains the SBMC model
train_sbmc:
	@python scripts/train.py \
		--checkpoint_dir $(OUTPUT)/emil/training_sbmc_theirs \
		--data $(OUTPUT)/emil/training_sequence_cornell/filelist.txt \
		--val_data $(OUTPUT)/emil/validation_sequence/filelist.txt \
		--env sbmc_ours --port 2001 --bs 1 --constant_spp \
		--spp 4

# Compares the two given models on a specified testset
# Outputs RMSE comparisons and visual comparisons
compare_models:
	@python scripts/compare_models.py \
		--model1 $(OUTPUT)/emil/trained_models/peters_all_cornell.pth \
		--model2 $(OUTPUT)/emil/trained_models/final_pretrained.pth \
		--save_dir $(OUTPUT)/emil/compare/img \
		--data $(OUTPUT)/emil/training_sequence/render_samples_seq \
		--amount 1

# -----------------------------------------------------------------------------

# The rest of this Makefiles demonstrates how to use the SBMC API and entry
# scripts for common tasks.

demo/render_samples: $(OUTPUT)/demo/test_samples/0000_0000.bin

# This demonstrates how we render .bin sample files for a test scene
$(OUTPUT)/demo/test_samples/0000_0000.bin: demo_data
	@python scripts/render_samples.py $(PBRT) \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/test_samples \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4 \
		--gt_spp 1

# This demonstrates how we render .bin sample files for a training dataset 
# using our random scene generator
demo/generate_scenes: $(OUTPUT)/demo/training_scenes/filelist.txt

$(OUTPUT)/demo/training_scenes/filelist.txt: demo_data
	@python scripts/generate_training_data.py $(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets $(OUTPUT)/demo/training_scenes --count 2 \
		--spp 4 --gt_spp 4 --height 128 --width 128
	@cd $(OUTPUT)/demo/training_scenes && find . -name "*.bin" > filelist.txt

# This shows how to use the visualization helper script to inspect the sample
# .bin files
demo/visualize:
	@python scripts/visualize_dataset.py $(OUTPUT)/demo/training_scenes \
		$(OUTPUT)/demo/dataviz --spp 1

# This demonstrates how to run pretrained models on .bin test scenes
demo/denoise: demo/render_samples pretrained_models
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/ours_4spp.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/gharbi2019_sbmc
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/bako2017_4spp.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/bako2017_finetuned

# This demonstrates how we render a .exr reference image for a test scene
demo/render_reference: demo_data
	@python scripts/render_exr.py $(PBRT) \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/reference/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4

# This demonstrates how we render .exr images for the comparison denoisers
# Rouselle2012 and Kalantari2015 require a GPU and are expected to fail on the
# CPU-only docker image.
demo/comparisons: demo/render_samples pretrained_models demo_data
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/comparisons/2017_bako_kpcn_finetuned/GITestSynthesizer_01.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/bako2017_finetuned
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/comparisons/ours/GITestSynthesizer_01.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/gharbi2019_sbmc
	@python scripts/render_exr.py $(SEN2011)/bin/pbrt \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/2011_sen_rpf/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4
	@python scripts/denoise_nfor.py $(BITTERLI2016)/build/denoiser \
		$(OUTPUT)/demo/test_samples \
		$(OUTPUT)/demo/comparisons/2016_bitterli_nfor/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --spp 4
	@python scripts/render_exr.py $(ROUSSELLE2012)/bin/pbrt \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/2012_rousselle_nlm/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4
	@python scripts/render_exr.py $(KALANTARI2015)/bin/pbrt \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/2015_kalantari_lbf/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4 \
		--kalantari2015_data $(KALANTARI2015)/pretrained/Weights.dat \
		$(KALANTARI2015)/pretrained/FeatureNorm.dat

# This demonstrates how to train a new model
# demo/train: demo/generate_scenes
# 	@python scripts/train.py \
# 		--checkpoint_dir $(OUTPUT)/demo/training \
# 		--data $(OUTPUT)/demo/training_scenes/filelist.txt \
# 		--env sbmc_ours --port 2001 --bs 1 \
# 		--spp 4

# This demonstrates how to train a baseline model (from [Bako 2017])
demo/train_kpcn: demo/generate_scenes
	@python scripts/train.py \
		--checkpoint_dir $(OUTPUT)/demo/training_kpcn \
		--data $(OUTPUT)/demo/training_scenes/filelist.txt \
		--constant_spp --env sbmc_kpcn --port 2001 --bs 1 \
		--kpcn_mode \
		--spp 4

# Launches a Visdom server to monitor the training
server:
	@python -m visdom.server -p 2001 &

demo/eval: precomputed_renderings
	@python scripts/compute_metrics.py data/renderings/ref output/eval.csv \
		--methods data/eval_methods.txt \
		--scenes data/eval_scenes.txt

# Download the data needed for the demo ---------------------------------------
demo_data: $(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt

pretrained_models: $(DATA)/pretrained_models/gharbi2019_sbmc/final.pth

test_scenes: $(DATA)/scenes/spaceship/scene.pbrt

precomputed_renderings: $(DATA)/renderings/4spp/spaceship.exr

$(DATA)/renderings/4spp/spaceship.exr:
	@echo "Downloading precomputed renderings from [Gharbi2019] (about 54 GB)"
	@python scripts/download.py $(REMOTE)/renderings.zip $(DATA)/renderings.zip
	cd $(DATA) && unzip renderings.zip
	rm $(DATA)/renderings.zip
	@python scripts/download.py $(REMOTE)/eval_methods.txt $(DATA)/eval_methods.txt
	@python scripts/download.py $(REMOTE)/eval_scenes.txt $(DATA)/eval_scenes.txt

$(DATA)/scenes/spaceship/scene.pbrt:
	@echo "Downloading test scenes (about 3 GB)"
	@python scripts/download.py $(REMOTE)/scenes.zip $(DATA)/scenes.zip
	cd $(DATA) && unzip scenes.zip
	rm $(DATA)/scenes.zip

$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt:
	@echo "Downloading demo data (about 30 MB)"
	@python scripts/download.py $(REMOTE)/demo.zip $(DATA)/demo.zip
	cd $(DATA) && unzip demo.zip
	rm $(DATA)/demo.zip

$(DATA)/pretrained_models/gharbi2019_sbmc/final.pth:
	@echo "Downloading pretrained models (about 512 MB)"
	@python scripts/download.py $(REMOTE)/pretrained_models.zip $(DATA)/pretrained_models.zip
	cd $(DATA) && unzip pretrained_models.zip
	rm $(DATA)/pretrained_models.zip
