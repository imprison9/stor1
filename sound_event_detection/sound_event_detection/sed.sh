# python run.py train_evaluate configs/Crnnplus.yaml 
# python evaluate.py --prediction experiments/Crnnplus/2024-06-06_12-35-06_28b75e8023be11ef88db5cff35c18da5/predictions.csv \
#                    --label data/eval/label.csv \
#                    --output result.txt
# python run.py train_evaluate configs/Crnn.yaml 
# python evaluate.py --prediction experiments/Crnn/2024-06-06_13-24-06_079fbcae23c511ef90495cff35c18da5/predictions.csv \
#                    --label data/eval/label.csv \
#                    --output result.txt
# python run.py train_evaluate configs/Crnn1.yaml 
# python evaluate.py --prediction experiments/Crnn1/2024-06-08_16-29-06_4069ac5e257111efba245cff35c1d408/predictions.csv \
#                    --label data/eval/label.csv \
#                    --output result.txt
# python run.py train_evaluate configs/Crnn2.yaml
python run.py train_evaluate configs/Crnn3.yaml 