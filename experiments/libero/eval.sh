# ECoT
python eval.py \
    --dataset libero_spatial \
    --num-trials-per-task 20

# ECoT with time-based frozen prompt control
python eval.py \
    --dataset libero_spatial \
    --num-trials-per-task 20 \
    --prompt-control-mode time \
    --frozen-prompt-max-freezing-time 5

# ECoT with uncertainty-based frozen prompt control (using far_mass_x_peak_distance as the uncertainty metric)
python eval.py \
    --dataset libero_spatial \
    --num-trials-per-task 20 \
    --prompt-control-mode metric_window_total_variation \
    --uncertainty-metric-name far_mass_x_peak_distance \
    --score-threshold 1.0 \
    --tv-window 5