
############### Table 5, Baseline 
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config baseline --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_baseline --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_baseline.pth

# generate maps (no need to convert because baseline is already a vanilla cnn)
python main.py --model pidinet --config baseline --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_baseline --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_baseline.pth

# 101 FPS
python throughput.py --model pidinet --config baseline --sa --dil -j 1 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



############### Table 5, PiDiNet
# train, or generate maps without conversion (uncomment the --evaluate)
python main.py --model pidinet --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS #--evaluate /path/to/table5_pidinet.pth

# generate maps with converted pidinet
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet.pth --evaluate-converted

# 96 FPS
python throughput.py --model pidinet_converted --config carv4 --sa --dil -j 1 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS



