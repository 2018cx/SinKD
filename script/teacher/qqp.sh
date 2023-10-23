python3 main_glue.py --do_lower_case \
					--do_train \
					--task_name qqp \
					--model_path $TEACHER_PATH \
					--per_gpu_batch_size 8 \
					--num_train_epochs 5 \
					--learning_rate 3e-6 \
                                        --seed 32