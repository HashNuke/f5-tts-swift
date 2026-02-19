import Foundation
import MLX
import MLXNN

public class F5DurationBlock: Module {
    let attn_norm: LayerNorm
    let attn: F5Attention
    let ff_norm: LayerNorm
    let ff: F5FeedForward

    init(dim: Int, heads: Int, dimHead: Int, ffMult: Int = 4, dropout: Float = 0.1) {
        self.attn_norm = LayerNorm(dimensions: dim)
        self.attn = F5Attention(dim: dim, heads: heads, dimHead: dimHead, dropout: dropout)
        self.ff_norm = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        self.ff = F5FeedForward(dim: dim, mult: ffMult, dropout: dropout, approximate: "tanh")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, rope: (MLXArray, Float)? = nil) -> MLXArray {
        let norm = attn_norm(x)
        let attnOutput = attn(norm, mask: mask, rope: rope)
        var output = x + attnOutput
        let normedOutput = ff_norm(output)
        let ffOutput = ff(normedOutput)
        output = output + ffOutput
        return output
    }
}
