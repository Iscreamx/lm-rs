use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, rms_norm, swiglu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            todo!("self_attention(...)");
            todo!("down_proj matmul and add residual");

            todo!("mlp(...)");
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        
        todo!("实现文本生成");
        
        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // score = Q @ K.T / sqrt(dim)
    let batch = n_kv_h * n_groups;
    for i in 0..batch {
        for a in 0..seq_len {
            for b in 0..total_seq_len {
                let mut score = 0.0;
                for c in 0..dqkv {
                    score += q.data()[i * dqkv + a * dqkv * n_groups * n_kv_h + c] * k.data()[i / n_groups * dqkv + b * dqkv * n_kv_h + c];
                }
                unsafe {
                    att_scores.data_mut()[i * seq_len * total_seq_len + a * total_seq_len + b] = score / (dqkv as f32).sqrt();
                }
            } 
        }
    }

    masked_softmax(att_scores);

    // attn_V = attn @ V
    for i in 0..batch {
        for a in 0..seq_len {
            for b in 0..dqkv {
                let mut attn_v = 0.0;
                for c in 0..total_seq_len {
                    attn_v += att_scores.data()[i * seq_len * total_seq_len + a * total_seq_len + c] * 
                        v.data()[i / n_groups * dqkv + c * dqkv * n_kv_h + b];
                }
                unsafe {
                    hidden_states.data_mut()[i * dqkv * seq_len + a * dqkv + b] = attn_v;
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // residual.print();
    rms_norm(hidden_states, residual, rms_w, eps);
    // hidden_states.print();
    matmul_transb(gate, 0., hidden_states, w_gate, 1.);
    // gate.print();
    matmul_transb(up, 0., hidden_states, w_up, 1.);
    // up.print();
    swiglu(up, gate);
    // up.print();
    matmul_transb(hidden_states, 0., up, w_down, 1.);
    // hidden_states.print();
    unsafe {
        residual
            .data_mut()
            .iter_mut()
            .zip(hidden_states.data())
            .for_each(|(r, &h)| *r += h);
    }
    // residual.print();
}

#[test]
pub fn test_self_attention_case_1() {
    let seq_len = 1;
    let total_seq_len = 4;
    let n_kv_h = 2;
    let n_groups = 2;
    let dqkv = 4;

    let q = Tensor::new(
        vec![
            -0.2386, -1.0934, 0.1558, 0.1750, -0.9526, -0.5442, 1.1985, 0.9604,
            -1.1074, -0.8403, -0.0020, 0.2240, 0.8766, -0.5379, -0.2994, 0.9785,
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );

    let k = Tensor::new(
        vec![
            1.5818, -0.2637, -0.8172, 1.4276, 1.7598, 1.1749, 0.2644, -0.6843,
            1.3014, -0.0108, -0.5931, 0.7040, -0.4759, -0.0982, 0.2107, -0.2471,
            -0.5589, 0.5258, -0.7888, 0.0871, -0.2525, 0.9114, -0.5615, 1.1892,
            -0.6095, -0.4202, -0.7433, 0.2429, 0.9471, 1.7863, -0.7762, -0.1504,
        ],
        &vec![total_seq_len, n_kv_h * dqkv],
    );

    let v = Tensor::new(
        vec![
            -1.6736, 0.4677, -0.6218, -0.6322, 0.0812, -0.3079, 0.7399, -0.6557,
            0.3716, 2.3700, -1.5539, -1.0817, -0.7842, 0.8656, -2.4925, -1.6823,
            -0.8224, 1.7161, -1.4522, 1.0167, -1.3481, 0.0612, 0.1985, 2.0584,
            0.4593, -0.3159, 0.1319, 2.0025, 1.0440, 1.1304, 0.1551, 0.9454,
        ],
        &vec![total_seq_len, n_kv_h * dqkv],
    );

    let mut hidden_states = Tensor::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);
    let mut att_scores = Tensor::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);

    self_attention(
        &mut hidden_states,
        &mut att_scores,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    let hidden_states_pt = Tensor::new(
        vec![
            -0.3546, 0.8697, -0.7388, 0.4540, -0.3181, 0.8330, -0.7203, 0.6573,
            -0.7169, 0.5333, -1.0760, -0.0938, -0.3117, 0.3560, -0.1350, 0.4386,
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );

    let att_scores_pt = Tensor::new(
        vec![
            0.2571, 0.2211, 0.1921, 0.3298, 0.2079, 0.1792, 0.2484, 0.3645,
            0.0789, 0.4878, 0.3315, 0.1017, 0.2618, 0.1728, 0.3293, 0.2361,
        ],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );

    println!("Rust Hidden States:");
    hidden_states.print();
    println!("PyTorch Hidden States:");
    hidden_states_pt.print();

    println!("Rust Attention Scores:");
    att_scores.print();
    println!("PyTorch Attention Scores:");
    att_scores_pt.print();

    assert!(hidden_states.close_to(&hidden_states_pt, 1e-3));
    assert!(att_scores.close_to(&att_scores_pt, 1e-3));        
}


#[test]
pub fn test_self_attention_case_2() {
    let seq_len = 2;
    let total_seq_len = 4;
    let n_kv_h = 2;
    let n_groups = 2;
    let dqkv = 4;

    let q = Tensor::new(
        vec![
            -0.2386, -1.0934, 0.1558, 0.1750, -0.9526, -0.5442, 1.1985, 0.9604,
            -1.1074, -0.8403, -0.0020, 0.2240, 0.8766, -0.5379, -0.2994, 0.9785,
            1.5818, -0.2637, -0.8172, 1.4276, 1.7598, 1.1749, 0.2644, -0.6843,
            1.3014, -0.0108, -0.5931, 0.7040, -0.4759, -0.0982, 0.2107, -0.2471,
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );

    let k = Tensor::new(
        vec![
            -0.5589, 0.5258, -0.7888, 0.0871, -0.2525, 0.9114, -0.5615, 1.1892,
            -0.6095, -0.4202, -0.7433, 0.2429, 0.9471, 1.7863, -0.7762, -0.1504,
            -1.6736, 0.4677, -0.6218, -0.6322, 0.0812, -0.3079, 0.7399, -0.6557,
            0.3716, 2.3700, -1.5539, -1.0817, -0.7842, 0.8656, -2.4925, -1.6823,
        ],
        &vec![total_seq_len, n_kv_h * dqkv],
    );

    let v = Tensor::new(
        vec![
            -0.8224, 1.7161, -1.4522, 1.0167, -1.3481, 0.0612, 0.1985, 2.0584,
            0.4593, -0.3159, 0.1319, 2.0025, 1.0440, 1.1304, 0.1551, 0.9454,
            0.2223, 0.3058, -0.3056, -2.4704, 0.9629, -0.9308, 1.0938, -1.0941,
            -0.0350, 0.7814, 0.1030, -1.2228, 0.3091, 0.4975, -1.0842, 1.1568,
        ],
        &vec![total_seq_len, n_kv_h * dqkv],
    );

    let mut hidden_states = Tensor::default(&vec![seq_len, n_kv_h * n_groups * dqkv]);
    let mut att_scores = Tensor::default(&vec![n_kv_h, n_groups, seq_len, total_seq_len]);

    self_attention(
        &mut hidden_states,
        &mut att_scores,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    let hidden_states_pt = Tensor::new(
        vec![
            0.049982, 0.42122, -0.37411, 0.32654, 0.037205, 0.44837, -0.41853, 0.11237,
            0.10639, -0.042732, 0.11702, 0.66082, -0.00083541, 0.20006, 0.24296, 0.97523,
            -0.075367, 0.63828, -0.37245, 0.48356, -0.071897, 0.79365, -0.054150, -0.92040,
            0.21796, 0.45082, 0.12352, 1.0362, 0.28217, 0.075252, 0.095889, 0.61944,
        ],
        &vec![seq_len, n_kv_h * n_groups * dqkv],
    );

    let att_scores_pt = Tensor::new(
        vec![
            0.2430, 0.4171, 0.2725, 0.0675, 0.2942, 0.3513, 0.0686, 0.2858,
            0.2526, 0.3707, 0.3414, 0.0354, 0.0907, 0.0474, 0.0430, 0.8189,
            0.2917, 0.0895, 0.3288, 0.2900, 0.2989, 0.4319, 0.1328, 0.1365,
            0.3975, 0.2850, 0.2131, 0.1044, 0.2190, 0.1819, 0.3094, 0.2898,
        ],
        &vec![n_kv_h, n_groups, seq_len, total_seq_len],
    );

    println!("Rust Hidden States:");
    hidden_states.print();
    println!("PyTorch Hidden States:");
    hidden_states_pt.print();

    println!("Rust Attention Scores:");
    att_scores.print();
    println!("PyTorch Attention Scores:");
    att_scores_pt.print();

    // assert!(hidden_states.close_to(&hidden_states_pt, 1e-3));
    // assert!(att_scores.close_to(&att_scores_pt, 1e-3));
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
