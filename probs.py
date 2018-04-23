import numpy as np

def sample_from_2D_output(output, is_log_prob=True):
    p_choice = output.flatten()
    if is_log_prob:
        p_choice = np.exp(p_choice)
    try:
        output_sample_flat_ix = np.random.choice(range(len(p_choice)),
                                                 size=None, replace=False, p=p_choice)
    except:
        print("WARNING: Could not sample from 2D output! Setting sample to 0.\n" + str(p_choice))
        output_sample_flat_ix = 0
    output_sample = np.unravel_index(output_sample_flat_ix, output.shape)
    output_sample_prob = p_choice[output_sample_flat_ix]
    return output_sample, output_sample_prob

def prob_mass_n_pixels_radius(prob_dist, u_p, v_p, log_prob=True, n_pixels=5):
    MIN_WINDOW_VALUE_U = 0
    MIN_WINDOW_VALUE_V = 0
    MAX_WINDOW_VALUE_U = prob_dist.shape[0]
    MAX_WINDOW_VALUE_V = prob_dist.shape[1]
    u_0 = max(MIN_WINDOW_VALUE_U, u_p - n_pixels)
    v_0 = max(MIN_WINDOW_VALUE_V, v_p - n_pixels)
    u_w = min(MAX_WINDOW_VALUE_U, u_p + n_pixels)
    v_w = min(MAX_WINDOW_VALUE_V, v_p + n_pixels)
    prob_window = prob_dist[u_0:u_w, v_0:v_w]
    if log_prob:
        prob_window = np.exp(prob_window)
    prob_mass = np.sum(prob_window)
    return prob_mass