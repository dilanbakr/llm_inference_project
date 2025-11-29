# llm_inference_project

# Low-Latency LLM Chat Service with Speculative Decoding

This project implements a **low-latency LLM inference service** that combines:

- A **small draft model** and a **larger target model** with **speculative decoding**
- **KV-cache** for efficient autoregressive decoding
- **FlashAttention-style attention kernels** (via `scaled_dot_product_attention`)
- Optionally, a **GQA (Grouped Query Attention)** model as the target for reduced KV memory

The goal is to demonstrate how modern LLM inference techniques can be combined to:
- **Reduce end-to-end latency**
- **Increase throughput**
- **Preserve answer quality close to a large teacher model**

---

## 1. Motivation

Naive LLM decoding with a large model is:

- **Slow**: each new token requires a full forward pass over the entire prefix.
- **Expensive**: attention is memory-bound and scales quadratically with sequence length.

This project explores how to build a **production-style chat service** that:

1. Uses a **small, fast draft model** to propose multiple candidate tokens.
2. Uses a **larger, higher-quality target model** to **accept or reject** the draft tokens in batches (speculative decoding).
3. Uses **KV-cache** so that past token representations are never recomputed.
4. Leverages **FlashAttention or fused attention kernels** to reduce memory I/O.
5. Optionally benefits from **GQA** in the target model to reduce KV-cache size.

---

## 2. High-Level Architecture

```text
            ┌────────────────────┐
            │      Client        │
            │  (Web / Backend)   │
            └────────┬───────────┘
                     │  /chat
                     ▼
            ┌────────────────────┐
            │   Chat Service     │
            │  (Fast API / gRPC) │
            └────────┬───────────┘
                     │
                     ▼
      ┌───────────────────────────────────────┐
      │     Speculative Decoding Orchestrator │
      │  - manages draft + target models      │
      │  - maintains KV-cache for both        │
      └───────┬───────────────────────┬───────┘
              │                       │
              ▼                       ▼
   ┌───────────────────┐     ┌────────────────────┐
   │   Draft Model     │     │   Target Model     │
   │ (small, fast)     │     │ (bigger, better)   │
   │ - KV-cache        │     │ - KV-cache         │
   │ - FlashAttention  │     │ - FlashAttention   │
   │                   │     │ - GQA (optional)   │
   └───────────────────┘     └────────────────────┘
