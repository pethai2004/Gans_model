import tensorflow as tf
## TODO : consider useing alpha with different value for each sample sets / each pixel
def grad_loss_0(D, alpha, p_r, p_g):
    p_g = tf.cast(p_g, dtype=tf.float32)
    p_r = tf.cast(p_r, dtype=tf.float32)
    interpolate = p_r + alpha * (p_g - p_r)
    
    with tf.name_scope("Gradient_Penalty"):
        with tf.GradientTape() as g_tape:
            g_tape.watch(interpolate)
            inter_pred_disc = D.forward_model(interpolate)
        g_grad = g_tape.gradient(inter_pred_disc, [interpolate])
        norm_g = tf.sqrt(tf.reduce_sum(tf.square(g_grad)))
        g_grad = tf.reduce_mean(tf.math.squared_difference(norm_g,  1))
    
    return g_grad
        
def generator_loss_0(z, G, D, train_sets):
    fake_img = G.forward_model(z)
#     real_score = D.forward_model(train_sets)
    fake_score = D.forward_model(fake_img)
    gen_loss = - tf.nn.softplus(fake_score) 
    
    return gen_loss

def discrim_loss_0(z, G, D, train_sets, g_penalty=0.):
    fake_img = G.forward_model(z)
    real_score = D.forward_model(train_sets)
    fake_score = D.forward_model(fake_img)
    disc_loss = tf.nn.softplus(fake_score) + tf.nn.softplus(- real_score) 
    
    if g_penalty > 0.:
        alpha_k = tf.random.normal((1, ), 0, 1)
        gd_Loss = grad_loss_0(D, alpha_k, p_r=train_sets, p_g=fake_img)
        disc_loss += gd_Loss
    else:
        gd_Loss = 0.
    return disc_loss, gd_Loss, (fake_score, real_score)

def generator_loss_wg(z, G, D, train_sets):
    fake_img = G.forward_model(z)
#     real_score = D.forward_model(train_sets)
    fake_score = D.forward_model(fake_img)
    gen_loss = - fake_score
    
    return gen_loss

def discrim_loss_wg(z, G, D, train_sets, g_penalty=0.):
    fake_img = G.forward_model(z)
    real_score = D.forward_model(train_sets)
    fake_score = D.forward_model(fake_img)
    disc_loss = fake_score - real_score
    
    if g_penalty > 0.:
        alpha_k = tf.random.normal((1, ), 0, 1)
        gd_Loss =  grad_loss_0(D, alpha_k, p_r=train_sets, p_g=fake_img)
        disc_loss += gd_Loss
    else:
        gd_Loss = 0.
    return disc_loss, gd_Loss, (fake_score, real_score)