python3 main_glue_distill.py --distill_loss kd+sinkhorn \
							--do_lower_case \
							--do_train \
							--task_name sst-2 \
    						        --teacher_path $TEACHER_PATH \
     						        --student_path $STUDENT_PATH \
							--per_gpu_batch_size 32 \
							--num_train_epochs 2 \
							--learning_rate 4e-5 \
							--alpha 0.9 \
							--temperature 2 \
							--beta 0.8 \
                                                        --seed 0