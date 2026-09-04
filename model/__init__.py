from .methods import MTP, STP, IMTP, IMTPs, MTNPKD


def get_model(config, device):
    if config.model == 'mtp':
        return MTP(config).to(device)
    elif config.model == 'stp':
        return STP(config).to(device)
    elif config.model == 'imtp':
        return IMTP(config).to(device)
    elif config.model == 'imtps':
        return IMTPs(config).to(device)
    elif config.model == 'mtnpkd':
        return MTNPKD(config).to(device)
    else:
        raise NotImplementedError
