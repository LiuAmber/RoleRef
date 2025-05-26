python generate_with_representation_control.py \
    --model_path "meta-llama/Llama-3.1-70B-Instruct" \
    --save_path "../../data/generate/representation/llama-0.05" \
    --safe_pattern_path "../../data/llama/{role}/rep_diff_half.pkl" \
    --ceoff -0.05