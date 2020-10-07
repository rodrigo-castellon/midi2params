"""
Not actually a script per se, but just some code that should be reused
once we start to *generate* with our model(s)
"""


# save a prediction example if we're on the right epoch
if epoch % config.logging.example_saving.save_every == 0 and i == 0:
    idx = config.logging.example_saving.i
    f0, loudness_db, pitches = batch['f0'], batch['loudness_db'], batch['pitches']
    cent_out = cent_out[idx].argmax(-1).float() - 50
    ld_out = ld_out[idx].argmax(-1).float() - 120
    pitches = pitches[idx].float()
    f0_pred, loudness_pred = get_predictions(cent_out, ld_out, pitches)
    # now plot these
    plt = plot_predictions_against_groundtruths(f0_pred, loudness_pred, f0[idx], loudness_db[idx])
    # give plt to wandb
    val_metrics['val_pred_plot'] = plt


def get_predictions(cent_out, ld_out, pitches):
    """
    Turn model outputs for a single example into actual continuous predictions
    we can compare easily to the ground truths.
    """
    
    # TODO: adapt for the multi-output model with absolute f0 prediction
    
    # compute predicted f0 from these predicted cents
    f0_pred = p2f(pitches.float()) * 2**(cent_out / 1200)

    return to_numpy(f0_pred), to_numpy(ld_out)  # ensure that we return numpy arrays

def plot_predictions_against_groundtruths(f0_pred, loudness_pred, f0, loudness):
    """
    Take model predictions and overlay ground truths with them in two vertically-aligned plots.
    Return `plt` to allow this to be sent to W&B.
    """
    # convert to numpy arrays if torch tensors
    f0_pred, loudness_pred, f0, loudness = to_numpy(f0_pred), to_numpy(loudness_pred), to_numpy(f0), to_numpy(loudness)
    
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(f0, linewidth=1, label='F0 (ground truth)')
    plt.plot(f0_pred, linewidth=1, label='F0 (predicted)')
    plt.title('F0 Comparison')
    plt.xlim(0, 1250)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loudness, label='Loudness (ground truth)')
    plt.plot(loudness_pred, label='Loudness (predicted)')
    plt.xlim(0, 1250)
    plt.title('Loudness Comparison')
    plt.legend()
    
    return plt