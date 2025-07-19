# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="OmniGen2/OmniGen2"
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--text_guidance_scale 5.0 \
--image_guidance_scale 3.0 \
--instruction "Change the background to swimming pool." \
--input_image_path example_images/blv.png \
--output_image_path outputs/output_edit.png \
--num_images_per_prompt 1