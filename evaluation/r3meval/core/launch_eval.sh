env_names=("hammer-v2-goal-observable" "drawer-open-v2-goal-observable" "bin-picking-v2-goal-observable" "assembly-v2-goal-observable" "button-press-topdown-v2-goal-observable")
# env_names=("kitchen_knob1_on-v3" "kitchen_light_on-v3" "kitchen_sdoor_open-v3" "kitchen_ldoor_open-v3" "kitchen_micro_open-v3")
seeds=("123" "124" "125")
embedding_names=("dinov2")
load_paths=("dinov2")
camera_names=("left_cap2" "right_cap2")
run_id=0

for env_name in ${env_names[@]} ; do
    for seed in ${seeds[@]} ; do
        for camera_name in ${camera_names[@]} ; do
            for embedding_name in ${embedding_names[@]} ; do
                for load_path in ${load_paths[@]} ; do
                    sbatch eval.sh ${env_name} ${seed} ${camera_name} ${embedding_name} ${load_path} ${run_id}
                done
            done
        done
    done
done