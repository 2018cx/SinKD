python3 main_glue.py --do_lower_case \
					--do_train \
					--task_name rte \
					--model_path $TEACHER_PATH \
					--per_gpu_batch_size 15 \
					--num_train_epochs 10 \
					--learning_rate 3e-5