python3 main_glue.py --do_lower_case \
					--do_train \
					--task_name cola \
					--model_path $TEACHER_PATH \
					--per_gpu_batch_size 2 \
					--num_train_epochs 8 \
					--learning_rate 2e-5