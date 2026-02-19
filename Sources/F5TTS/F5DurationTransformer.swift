import Foundation
import MLX
import MLXNN
import MLXRandom

public class F5DurationTransformer: Module {
    let dim: Int
    let text_embed: F5TextEmbedding
    let input_embed: F5DurationInputEmbedding
    let rotary_embed: F5RotaryEmbedding
    let transformer_blocks: [F5DurationBlock]
    let norm_out: RMSNorm
    let depth: Int

    init(
        dim: Int,
        depth: Int = 8,
        heads: Int = 8,
        dimHead: Int = 64,
        dropout: Float = 0.0,
        ffMult: Int = 4,
        melDim: Int = 100,
        textNumEmbeds: Int = 256,
        textDim: Int? = nil,
        convLayers: Int = 0
    ) {
        self.dim = dim
        let actualTextDim = textDim ?? melDim
        self.text_embed = F5TextEmbedding(textNumEmbeds: textNumEmbeds, textDim: actualTextDim, convLayers: convLayers)
        self.input_embed = F5DurationInputEmbedding(melDim: melDim, textDim: actualTextDim, outDim: dim)
        self.rotary_embed = F5RotaryEmbedding(dim: dimHead)
        self.depth = depth

        self.transformer_blocks = (0 ..< depth).map { _ in
            F5DurationBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: ffMult, dropout: dropout)
        }

        self.norm_out = RMSNorm(dimensions: dim)
    }

    func callAsFunction(
        cond: MLXArray,
        text: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let seqLen = cond.shape[1]

        let textEmbed = text_embed(text, seqLen: seqLen)
        var x = input_embed(cond: cond, textEmbed: textEmbed)

        let rope = rotary_embed.forwardFromSeqLen(seqLen)

        for block in transformer_blocks {
            x = block(x, mask: mask, rope: rope)
        }

        return norm_out(x)
    }
}
