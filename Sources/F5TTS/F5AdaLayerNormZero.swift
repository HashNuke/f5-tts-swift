import Foundation
import MLX
import MLXNN
import MLXRandom

// AdaLayerNormZero
// return with modulated x for attn input, and params for later mlp modulation

class F5AdaLayerNormZero: Module {
    let silu: SiLU
    let linear: Linear
    let norm: LayerNorm

    init(dim: Int) {
        self.silu = SiLU()
        self.linear = Linear(dim, dim * 6)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let embProcessed = linear(silu(emb))
        let parts = embProcessed.split(parts: 6, axis: 1)
        let shiftMsa = parts[0]
        let scaleMsa = parts[1]
        let gateMsa = parts[2]
        let shiftMlp = parts[3]
        let scaleMlp = parts[4]
        let gateMlp = parts[5]

        let normX = norm(x)
        let modulatedX = normX * (MLXArray(1) + MLX.expandedDimensions(scaleMsa, axis: 1)) + MLX.expandedDimensions(shiftMsa, axis: 1)
        return (modulatedX, gateMsa, shiftMlp, scaleMlp, gateMlp)
    }
}
