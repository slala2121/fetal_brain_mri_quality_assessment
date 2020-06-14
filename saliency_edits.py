
"""
Changes to the keras-vis/vis/visualization/saliency.py 

Augmented to return additional data useful for saliency visualization

"""

def visualize_cam_with_losses(input_tensor, losses,
                              seed_input, penultimate_layer,
                              grad_modifier=None):
    """Generates a gradient based class activation map (CAM) by using positive gradients of `input_tensor`
    with respect to weighted `losses`.

    For details on grad-CAM, see the paper:
    [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization]
    (https://arxiv.org/pdf/1610.02391v1.pdf).

    Unlike [class activation mapping](https://arxiv.org/pdf/1512.04150v1.pdf), which requires minor changes to
    network architecture in some instances, grad-CAM has a more general applicability.

    Compared to saliency maps, grad-CAM is class discriminative; i.e., the 'cat' explanation exclusively highlights
    cat regions and not the 'dog' region and vice-versa.

    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model input for which activation map needs to be visualized.
        penultimate_layer: The pre-layer to `layer_idx` whose feature maps should be used to compute gradients
            with respect to filter output.
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
            specify anything, gradients are unchanged (Default value = None)

    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards minimizing the
        weighted `losses`.
    """
    penultimate_output = penultimate_layer.output
    opt = Optimizer(input_tensor, losses, wrt_tensor=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = grads / (np.max(grads) + K.epsilon())

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output.
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    # Generate heatmap by computing weight * output over feature maps
    output_dims = utils.get_img_shape(penultimate_output)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights): 
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    
    
    # ReLU thresholding to exclude pattern mismatch information (negative gradients).
    #unfiltered_heatmap=heatmap.copy()
    raw_heatmap=heatmap.copy()
    neg_heatmap=heatmap*-1.0
    neg_heatmap=np.maximum(neg_heatmap,0)
    heatmap = np.maximum(heatmap, 0)

    # The penultimate feature map size is definitely smaller than input image.
    input_dims = utils.get_img_shape(input_tensor)[2:]
    heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')
    neg_heatmap=imresize(neg_heatmap,input_dims,interp='bicubic',mode='F')
    raw_heatmap_resized=imresize(raw_heatmap,input_dims,interp='bicubic',mode='F')
    # Normalize and create heatmap.
    heatmap = utils.normalize(heatmap)
    neg_heatmap=utils.normalize(neg_heatmap)
    # raw_heatmap_resized=utils.normalize(raw_heatmap_resized)
    
    #print("running updated smap method")
   #return np.uint8(cm.jet(heatmap)[..., :3] * 255)
    return heatmap,neg_heatmap,raw_heatmap,raw_heatmap_resized, penultimate_output_value
