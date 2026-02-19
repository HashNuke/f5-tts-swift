import Foundation
import MLX
import MLXNN
import MLXRandom

// AdaLayerNormZero for final layer
// return only with modulated x for attn input, cuz no more mlp modulation

class F5AdaLayerNormZero_Final: Module {
    let silu: SiLU
    let linear: Linear
    let norm: LayerNorm

    init(dim: Int) {
        self.silu = SiLU()
        self.linear = Linear(dim, dim * 2)
        self.norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray? = nil) -> MLXArray {
        guard let emb = emb else {
            fatalError("Embedding tensor must not be nil")
        }

        let embProcessed = linear(silu(emb))

        let scaleAndShift = embProcessed.split(parts: 2, axis: 1)
        let scale = scaleAndShift[0]
        let shift = scaleAndShift[1]

        let modulatedX = norm(x) * (MLXArray(1) + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)

        return modulatedX
    }
}
