from utils import DefaultConfig, pre_build_G, pre_build_D

EPOCHS = 300

TrainerConfigs = []
DiscriminatorConfigs = []
GeneratorConfigs = []

test_train_config = DefaultConfig(epochs=5, clip_norm=None, switch_ratio={"G" : 2, "D" : 1}, img_size=(16, 16), targ_img_size=(16, 16), 
                                n_gpu=["gpu:0", "gpu:1"], G_lr=0.001, D_lr=0.001, max_G_lr=0.001, max_D_lr=0.001, optimizer="RMSprop",
                                lr_schedual="linear", latent_z_dim=200, train_ratio=None, grad_penalty=2, applied_D_method="resize",
                                applied_G_method=None, seed=100)

test_g_config = {"BaseFilters" : [10, 10 ,10, 10], "filters" : ([10, 10, 10], [10, 10, 10]),
        "out_units" : 10, "add_noise" : True, "dense_act" : "selu", "conv_act" : "selu", 
        "out_act" : "selu", "num_layers" : 2, "dense_units" : 20100, "kernel_size" : (2, 2)}

test_d_config = {"filters" : [(10, 2), (10, 2), (10, 2), (10, 2), (10, 2)] ,
                "out_units" : 10, "units_dense" : [10 , 10], "add_noise" : True, "conv_act" : "selu",
                "dense_act" : "selu", "out_act" : "selu", "num_layers" : 2, "kernel_size" : (2, 2), "dense_units" : 10}
########################################################################################################
########################################################################################################
########################################################################################################

train_config001 = DefaultConfig(epochs=300, clip_norm=None, switch_ratio={"G" : 1, "D" : 1}, img_size=(32, 32), targ_img_size=(32, 32), 
                                n_gpu=["gpu:0", "gpu:1"], G_lr=0.0001, D_lr=0.0001, max_G_lr=0.0001, max_D_lr=0.0001, optimizer="RMSprop",
                                lr_schedual=None, latent_z_dim=200, train_ratio=None, grad_penalty=0.1, applied_D_method=None,
                                applied_G_method=None, seed=100)

g_config001 = {"BaseFilters" : [300, 300 ,300, 300, 300, 300], "filters" : ([300, 300, 200], [300, 300, 200]),
        "out_units" : 300, "add_noise" : True, "dense_act" : "selu", "conv_act" : "selu", 
        "out_act" : "selu", "num_layers" : 6, "dense_units" : 200, "kernel_size" : (5, 5)}

d_config001 = {"filters" : [(200, 2), (300, 2), (300, 2), (300, 2), (300, 2), (300, 2), (300, 2)] ,
                "out_units" : 300, "units_dense" : [300 , 300], "add_noise" : True, "conv_act" : "selu",
                "dense_act" : "selu", "out_act" : "selu", "num_layers" : 6, "kernel_size" : (5, 5), "dense_units" : 200}

TrainerConfigs.append(train_config001)
DiscriminatorConfigs.append(d_config001)
GeneratorConfigs.append(g_config001)

########################################################################################################
train_config002 = DefaultConfig(epochs=300, clip_norm=None, switch_ratio={"G" : 1, "D" : 1}, img_size=(32, 32), targ_img_size=(32, 32), 
                                n_gpu=["gpu:0", "gpu:1"], G_lr=0.001, D_lr=0.001, max_G_lr=0.001, max_D_lr=0.001, optimizer="RMSprop",
                                lr_schedual="linear", latent_z_dim=200, train_ratio=None, grad_penalty=2, applied_D_method="resize",
                                applied_G_method=None, seed=100)

g_config002 = {"BaseFilters" : [300, 400 ,500, 400, 300, 300], "filters" : ([300, 300, 200], [300, 300, 200]),
        "out_units" : 300, "add_noise" : True, "dense_act" : "selu", "conv_act" : "selu", 
        "out_act" : "selu", "num_layers" : 6, "dense_units" : 200, "kernel_size" : (5, 5)}

d_config002 = {"filters" : [(200, 2), (300, 2), (300, 2), (200, 2), (300, 2), (300, 2), (300, 2)] ,
                "out_units" : 300, "units_dense" : [300 , 300], "add_noise" : False, "conv_act" : "relu",
                "dense_act" : "selu", "out_act" : "selu", "num_layers" : 6, "kernel_size" : (5, 5), "dense_units" : 200}

TrainerConfigs.append(train_config002)
DiscriminatorConfigs.append(d_config002)
GeneratorConfigs.append(g_config002)

########################################################################################################
train_config003 = DefaultConfig(epochs=300, clip_norm=None, switch_ratio={"G" : 1, "D" : 1}, img_size=(16, 16), targ_img_size=(16, 16), 
                                n_gpu=["gpu:0", "gpu:1"], G_lr=0.001, D_lr=0.001, max_G_lr=0.001, max_D_lr=0.001, optimizer="RMSprop",
                                lr_schedual="linear", latent_z_dim=200, train_ratio=None, grad_penalty=2, applied_D_method="resize",
                                applied_G_method=None, seed=100)

g_config003 = {"BaseFilters" : [300, 400 ,500, 400, 300, 300], "filters" : ([300, 300, 200], [300, 300, 200]),
        "out_units" : 300, "add_noise" : True, "dense_act" : "selu", "conv_act" : "selu", 
        "out_act" : "selu", "num_layers" : 6, "dense_units" : 200, "kernel_size" : (5, 5)}

d_config003 = {"filters" : [(200, 2), (300, 2), (300, 2), (200, 2), (300, 2), (300, 2), (300, 2)] ,
                "out_units" : 300, "units_dense" : [300 , 300], "add_noise" : False, "conv_act" : "relu",
                "dense_act" : "selu", "out_act" : "selu", "num_layers" : 6, "kernel_size" : (5, 5), "dense_units" : 200}

TrainerConfigs.append(train_config003)
DiscriminatorConfigs.append(d_config002)
GeneratorConfigs.append(g_config003)
########################################################################################################

train_config004 = DefaultConfig(epochs=300, clip_norm=None, switch_ratio={"G" : 1, "D" : 1}, img_size=(8, 8), targ_img_size=(8, 8), 
                                n_gpu=["gpu:0", "gpu:1"], G_lr=0.001, D_lr=0.001, max_G_lr=0.001, max_D_lr=0.001, optimizer="RMSprop",
                                lr_schedual="linear", latent_z_dim=200, train_ratio=None, grad_penalty=2, applied_D_method="resize",
                                applied_G_method=None, seed=100)

g_config003 = {"BaseFilters" : [300, 400 ,500, 400, 300, 300], "filters" : ([300, 300, 200], [300, 300, 200]),
        "out_units" : 300, "add_noise" : True, "dense_act" : "selu", "conv_act" : "selu", 
        "out_act" : "selu", "num_layers" : 6, "dense_units" : 200, "kernel_size" : (5, 5)}

d_config003 = {"filters" : [(200, 2), (300, 2), (300, 2), (200, 2), (300, 2), (300, 2), (300, 2)] ,
                "out_units" : 300, "units_dense" : [300 , 300], "add_noise" : False, "conv_act" : "relu",
                "dense_act" : "selu", "out_act" : "selu", "num_layers" : 6, "kernel_size" : (5, 5), "dense_units" : 200}

TrainerConfigs.append(train_config003)
DiscriminatorConfigs.append(d_config002)
GeneratorConfigs.append(g_config003)
########################################################################################################

train_config005 = DefaultConfig(epochs=300, clip_norm=None, switch_ratio={"G" : 1, "D" : 1}, img_size=(64, 64), targ_img_size=(64, 64), 
                                n_gpu=["gpu:0", "gpu:1"], G_lr=0.001, D_lr=0.001, max_G_lr=0.001, max_D_lr=0.001, optimizer="RMSprop",
                                lr_schedual="linear", latent_z_dim=200, train_ratio=None, grad_penalty=2, applied_D_method="resize",
                                applied_G_method=None, seed=100)

g_config003 = {"BaseFilters" : [300, 400 ,500, 400, 300, 300], "filters" : ([300, 300, 200], [300, 300, 200]),
        "out_units" : 300, "add_noise" : True, "dense_act" : "selu", "conv_act" : "selu", 
        "out_act" : "selu", "num_layers" : 6, "dense_units" : 200, "kernel_size" : (5, 5)}

d_config003 = {"filters" : [(200, 2), (300, 2), (300, 2), (200, 2), (300, 2), (300, 2), (300, 2)] ,
                "out_units" : 300, "units_dense" : [300 , 300], "add_noise" : False, "conv_act" : "relu",
                "dense_act" : "selu", "out_act" : "selu", "num_layers" : 6, "kernel_size" : (5, 5), "dense_units" : 200}

TrainerConfigs.append(train_config003)
DiscriminatorConfigs.append(d_config002)
GeneratorConfigs.append(g_config003)
########################################################################################################