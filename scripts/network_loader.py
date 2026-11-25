"""Loader for the shared network module used by the training and analysis scripts."""


def load_q1_module(*_args):
    import bp_networks as q1
    return q1
