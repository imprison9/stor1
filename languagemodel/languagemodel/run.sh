#6 python main.py --cuda --epochs 40 --model LSTM --lr 5 --save lstm1.pt
#7 python main.py --cuda --epochs 40 --model Transformer --lr 5 --save Transformer.pt
#1 python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           model.pt
#2 python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    model.pt
#3 python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied --save LSTM.pt
#4 python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --model GRU --tied --save GRU.pt
#5 python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --model RNN_RELU --tied --save RNN_RELU.pt

# python generate.py --data ./data/gigaspeech --cuda --checkpoint ./model.pt --outf model.txt
# python generate.py --data ./data/gigaspeech --cuda --checkpoint ./GRU.pt --outf GRU.txt
# python generate.py --data ./data/gigaspeech --cuda --checkpoint ./LSTM.pt --outf LSTM.txt
# python generate.py --data ./data/gigaspeech --cuda --checkpoint ./RNN_RELU.pt --outf RNN_RELU.txt 因为loss爆炸（relu）nan等问题
# python generate.py --data ./data/gigaspeech --cuda --checkpoint ./lstm1.pt --outf lstm1.txt
# python generate.py --data ./data/gigaspeech --cuda --checkpoint ./Transformer.pt --outf Transformer.txt 
#8 python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --model RNN_TANH --tied --save RNN_TANH.pt
python generate.py --data ./data/gigaspeech --cuda --checkpoint ./RNN_TANH.pt --outf RNN_TANH.txt