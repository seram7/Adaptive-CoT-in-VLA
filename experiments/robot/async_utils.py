import asyncio
from uuid import uuid4
import vllm
import torch

async def engine_inference(
    model,
    engine,
    input_ids = None,
    pixel_values = None,
    sampling_params = None,
):
    # Visual Feature Extraction (shared across batched lanuaged inputs)
    patch_features = model.vision_backbone(pixel_values)
    projected_patch_embeddings = model.projector(patch_features)
    embds = model.input_embds
    input_embeddings = [embds(ids) for ids in input_ids]
    
    # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
    multimodal_embeddings = [torch.cat([inemb[:, :1, :], projected_patch_embeddings, inemb[:, 1:, :]], dim=1).squeeze(0) for inemb in input_embeddings]#[0] 
    prompt = [[32000] * emb.shape[-2] for emb in multimodal_embeddings]
    inputs = [{"prompt_token_ids": p, "multi_modal_data": {"image":m}} for p, m in zip(prompt, multimodal_embeddings)]
    tasks = [asyncio.create_task(run_query(vllm.inputs.TokensPrompt(**q), engine, sampling_params)) for q in inputs]
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
    return results

async def run_query(query, engine, params):
    request_id = uuid4()
    outputs = engine.generate(query, params, request_id)
    async for output in outputs:
        final_output = output
    responses = []
    for output in final_output.outputs:
        responses.append(output.text)
    return responses