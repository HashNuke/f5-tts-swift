import Foundation
import MLX
import MLXNN
import MLXRandom

// DiT block

class F5DiTBlock: Module {
    let attn_norm: F5AdaLayerNormZero
    let attn: F5Attention
    let ff_norm: LayerNorm
    let ff: F5FeedForward

    init(dim: Int, heads: Int, dimHead: Int, ffMult: Int = 4, dropout: Float = 0.1) {
        self.attn_norm = F5AdaLayerNormZero(dim: dim)
        self.attn = F5Attention(dim: dim, heads: heads, dimHead: dimHead, dropout: dropout)
        self.ff_norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.ff = F5FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh")
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil) -> MLXArray {
        let (norm, gateMsa, shiftMlp, scaleMlp, gateMlp) = attn_norm(x, emb: t)
        let attnOutput = attn(norm, mask: mask, rope: rope)
        var output = x + gateMsa.expandedDimensions(axis: 1) * attnOutput
        let normedOutput = ff_norm(output) * (1 + scaleMlp.expandedDimensions(axis: 1)) + shiftMlp.expandedDimensions(axis: 1)
        let ffOutput = ff(normedOutput)
        output = output + MLX.expandedDimensions(gateMlp, axis: 1) * ffOutput
        return output
    }
}
