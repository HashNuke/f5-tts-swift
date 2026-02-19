import Foundation
import MLX
import MLXNN
import MLXRandom

class F5InputEmbedding: Module {
    let proj: Linear
    let conv_pos_embed: F5ConvPositionEmbedding

    init(melDim: Int, textDim: Int, outDim: Int) {
        self.proj = Linear(melDim * 2 + textDim, outDim)
        self.conv_pos_embed = F5ConvPositionEmbedding(dim: outDim)
    }

    func callAsFunction(
        x: MLXArray,
        cond: MLXArray,
        textEmbed: MLXArray,
        dropAudioCond: Bool = false
    ) -> MLXArray {
        var cond = cond
        if dropAudioCond {
            cond = MLX.zeros(like: cond)
        }

        let combined = MLX.concatenated([x, cond, textEmbed], axis: -1)
        var output = proj(combined)
        output = conv_pos_embed(output) + output
        return output
    }
}
