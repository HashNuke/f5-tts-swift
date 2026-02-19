import Foundation
import MLX
import MLXNN
import MLXRandom

// Transformer backbone using DiT blocks

public class F5DiT: Module {
    let dim: Int
    let time_embed: F5TimestepEmbedding
    let text_embed:F5TextEmbedding
    let input_embed: F5InputEmbedding
    let rotary_embed: F5RotaryEmbedding
    let transformer_blocks: [F5DiTBlock]
    let norm_out: F5AdaLayerNormZero_Final
    let proj_out: Linear
    let depth: Int

    init(
        dim: Int,
        depth: Int = 8,
        heads: Int = 8,
        dimHead: Int = 64,
        dropout: Float = 0.1,
        ffMult: Int = 4,
        melDim: Int = 100,
        textNumEmbeds: Int = 256,
        textDim: Int? = nil,
        convLayers: Int = 0
    ) {
        self.dim = dim
        let actualTextDim = textDim ?? melDim
        self.time_embed = F5TimestepEmbedding(dim: dim)
        self.text_embed = F5TextEmbedding(textNumEmbeds: textNumEmbeds, textDim: actualTextDim, convLayers: convLayers)
        self.input_embed = F5InputEmbedding(melDim: melDim, textDim: actualTextDim, outDim: dim)
        self.rotary_embed = F5RotaryEmbedding(dim: dimHead)
        self.depth = depth

        self.transformer_blocks = (0 ..< depth).map { _ in
            F5DiTBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: ffMult, dropout: dropout)
        }

        self.norm_out = F5AdaLayerNormZero_Final(dim: dim)
        self.proj_out = Linear(dim, melDim)
    }

    func callAsFunction(
        x: MLXArray,
        cond: MLXArray,
        text: MLXArray,
        time: MLXArray,
        dropAudioCond: Bool,
        dropText: Bool,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]

        let time = (time.ndim == 0) ? MLX.repeated(time.expandedDimensions(axis: 0), count: batchSize, axis: 0) : time
        let t = time_embed(time)
        let textEmbed = text_embed(text, seqLen: seqLen, dropText: dropText)
        var x = input_embed(x: x, cond: cond, textEmbed: textEmbed, dropAudioCond: dropAudioCond)

        let rope = rotary_embed.forwardFromSeqLen(seqLen)

        for block in transformer_blocks {
            x = block(x, t: t, mask: mask, rope: rope)
        }

        x = norm_out(x, emb: t)
        return proj_out(x)
    }
}
