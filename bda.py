import sys
sys.path.append('C:/Users/user/Informer2020/')
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch

args = dotdict()

args.model = 'informer' # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = 'ETTh1' # data
args.root_path = 'C:/Users/user/Informer2020/' # root path of data file
args.data_path = 'Informer_data.csv' # data file
args.features = 'M' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'OT' # target feature in S or MS task
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = 'C:/Users/user/Informer2020/check/' # location of model checkpoints

## len에 대한 정리 : seq_len과 label_len, pred_len은 20,20,1로 훈련 시에 사용한다. 이 후에, prediction할 때, 1개의 값이 도출되면 원래 데이터에 이업

args.seq_len = 20 # input sequence length of Informer encoder, 40으로 설정하는 이유는 궁극적인 목표 자체가 20개의 데이터로 1개의 데이터를 만들어서 총 20개의 내가 예측한 데이터를 얻는 것이 목적이므로.
args.label_len = 20 # start token length of Informer decoder, label_len만큼 이전 데이터를 참고
args.pred_len = 1 # prediction sequence length, 예측하는 길이 / 만약에 20이라고 설정해두면, 20개의 sequence로 뒤의 20개 전체의 pattern을 학습하는건가? 그러면 차라리 1개씩 하는게 낫나?
# pred는 1로 하는 게 맞다. 
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
# decoder는 output의 결과가 [seq_len - label_len:seq_len + pred_len]으로 나온다.

args.enc_in = 14 # encoder input size
args.dec_in = 14 # decoder input size
args.c_out = 14 # output size
args.factor = 5 # probsparse attn factor
args.d_model = 512 # dimension of model
args.n_heads = 8 # num of heads
args.e_layers = 2 # num of encoder layers
args.d_layers = 1 # num of decoder layers
args.d_ff = 2048 # dimension of fcn in model
args.dropout = 0.05 # dropout
args.attn = 'prob' # attention used in encoder, options:[prob, full]
args.embed = 'timeF' # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu' # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False # whether to output attention in encoder
args.mix = True
args.padding = 0
args.freq = 'h'
args.inverse = False # output을 원래 형태로 돌려놓을 것인가? -> True / default값은 False임.


args.batch_size = 16 
args.learning_rate = 0.0001
args.loss = 'mse' # feature 예측에는 mse를 사용하는 게 나을려나?
args.lradj = 'type1'
args.use_amp = False # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 6
args.patience = 3
args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Set augments by using data name
data_parser = {
    'ETTh1':{'data':'Informer_data.csv','T':'OT','M':[14,14,14],'S':[1,1,1],'MS':[14,14,1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii)

    # set experiments
    exp = Exp(args)
    
    # train
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    # test
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()