python3 main_glue.py --do_lower_case \
					--do_train \
					--task_name mnli \
					--model_path $TEACHER_PATH \
					--per_gpu_batch_size 128 \
					--num_train_epochs 7 \
					--learning_rate 1e-5 