import argparse


def parse():
    p = argparse.ArgumentParser("UniGNN: Unified Graph and Hypergraph Message Passing Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='cocitation', help='data name (coauthorship/cocitation)')
    p.add_argument('--dataset', type=str, default='citeseer', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation)')
    p.add_argument('--model_name', type=str, default='UniGCNII', help='UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)')
    p.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')#mean
    p.add_argument('--second-aggregate', type=str, default='sum', help='aggregation for node x_i: max, sum, mean')
    p.add_argument('--add-self-loop', action="store_true", default=True, help='add-self-loop to hypergraph')
    p.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
    p.add_argument('--activation', type=str, default='relu', help='activation layer between UniConvs')
    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=64, help='number of hidden features, note that actually it\'s #nhid x #nhead')
    p.add_argument('--nhead', type=int, default=8, help='number of conv heads')
    p.add_argument('--dropout', type=float, default=0.2, help='dropout probability after UniConv layer')
    p.add_argument('--input-drop', type=float, default=0.6, help='dropout probability for input layer')
    p.add_argument('--attn-drop', type=float, default=0.6, help='dropout probability for attentions in UniGATConv')
    p.add_argument('--lr', type=float, default=0.01, help='learning rate')
    p.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    p.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    p.add_argument('--n-runs', type=int, default=10, help='number of runs for repeated experiments')
    p.add_argument('--gpu', type=int, default=2, help='gpu id to use')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--patience', type=int, default=200, help='early stop after specific epochs')
    p.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')
    p.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')
    p.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')


    p.add_argument('--save_file', default='results.csv', help='save file name')

    #argument for batch_norm
    p.add_argument("--type_norm", default="None",help="The type of the norm.")
    p.add_argument('--num_groups', type=int, default=10, help='The number of groups.')
    p.add_argument('--skip_weight', type=float, default=0.0001, help='skip_weight.')
    p.add_argument('--skipweight_learnable', action='store_true', default=False, help='learnable  skip_weight')
    p.add_argument('--wd_sw', type=float, default=1e-6, help='weight decay (L2 loss on parameters).')
    p.add_argument('--lr_sw', type=float, default=5e-4,help='Initial learning rate.')
    p.add_argument('--multiple', type=float, default=1, help='the multiple of pairnorm')
    p.add_argument('--mul_learnable', action='store_true', default=False, help='learnable  multiple')
    p.add_argument('--wd_mul', type=float, default=1e-5, help='weight decay (L2 loss on parameters).')
    p.add_argument('--lr_mul', type=float, default=1e-7,help='Initial learning rate.')


    p.add_argument('--alpha_learnable', action='store_true', default=True, help='l')

    return p.parse_args()
