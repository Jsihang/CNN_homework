from copy import deepcopy


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self):
        return self.ema_model.state_dict()
