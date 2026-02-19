import Foundation
import MLX
import MLXFast
import MLXNN

class F5DurationInputEmbedding: Module {
    let proj: Linear
    let conv_pos_embed: F5ConvPositionEmbedding

    init(melDim: Int, textDim: Int, outDim: Int) {
        self.proj = Linear(melDim + textDim, outDim)
        self.conv_pos_embed = F5ConvPositionEmbedding(dim: outDim)
        super.init()
    }

    func callAsFunction(
        cond: MLXArray,
        textEmbed: MLXArray
    ) -> MLXArray {
        var output = proj(MLX.concatenated([cond, textEmbed], axis: -1))
        output = conv_pos_embed(output) + output
        return output
    }
}
