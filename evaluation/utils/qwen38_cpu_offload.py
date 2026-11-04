"""Qwen38 cpu offload."""


def embedding_cpu_device_map(model_class, config, max_memory):
    import torch
    from accelerate import init_empty_weights
    from transformers.integrations.accelerate import infer_auto_device_map

    with init_empty_weights():
        model = model_class._from_config(config, dtype=torch.bfloat16)
    embeddings = {name: module for name, module in model.named_modules()
                  if name.endswith('.ple.ple_embedding.ngram_embedding')}
    if not embeddings:
        raise RuntimeError('Expected Qwen n-gram embedding was not found')
    lookup_bytes = sum(p.numel() * p.element_size()
                       for module in embeddings.values() for p in module.parameters())
    remaining = dict(max_memory)
    remaining['cpu'] -= lookup_bytes
    if remaining['cpu'] <= 0:
        raise RuntimeError('CPU budget cannot hold the n-gram embedding')


    for name in embeddings:
        model.set_submodule(name, torch.nn.Identity())
    mapping = dict(infer_auto_device_map(
        model, max_memory=remaining, no_split_module_classes=model._no_split_modules))


    for name in embeddings:
        parent = name.rsplit('.ple.ple_embedding.', 1)[0]
        mapping = {key: value for key, value in mapping.items()
                   if key != parent and not key.startswith(parent + '.')}
        mapping[parent] = 'cpu'
    if any(str(device) in ('disk', 'meta') for device in mapping.values()):
        raise RuntimeError('CPU and GPU budgets cannot hold original-precision weights')
    if not any(str(device) not in ('cpu', 'disk', 'meta') for device in mapping.values()):
        raise RuntimeError('CPU offload requires some layers on GPU')
    return mapping


def keep_ngram_lookup_on_cpu(model):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

    for name, module in model.named_modules():
        if not name.endswith('.ple.ple_embedding.ngram_embedding'):
            continue

        remove_hook_from_module(module)
        add_hook_to_module(module, AlignDevicesHook(
            execution_device='cpu', offload=False, io_same_device=True))
        print(f'CPU embedding lookup: {name}; only selected rows return to GPU', flush=True)
