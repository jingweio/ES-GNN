import os


class args:
    method = "esgnn"

    dataset = "chameleon"  # 'chameleon', 'squirrel', 'film', 'twitch-e', 'polblogs', 'etg_syn_hom'
    sub_dataset = ""  # 'DE', "1.0", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0"
    DATAPATH = os.getcwd() + "/data/"

    rnd_seed = 0
    label_rate = 60
    split_index = -1

    hidden_channels = 64
    reg = 6e-6
    lr = 0.06
    num_layers = 2
    dropout = 0.1
    re_eps = 0.1
    ir_eps = 0.3
    ir_con_lambda = 3e-6
    nepoch = 1000
    early = 100

    cpu = False
    rocauc = False  # set the eval function to rocauc
