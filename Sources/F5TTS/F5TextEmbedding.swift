import Foundation
import MLX
import MLXNN

class F5TextEmbedding: Module {
    let text_embed: Embedding
    var extraModeling: Bool = false
    var precomputeMaxPos: Int = 4096
    var freqsCis: MLXArray?
    var text_blocks: Sequential?

    init(textNumEmbeds: Int, textDim: Int, convLayers: Int = 0, convMult: Int = 2) {
        self.text_embed = Embedding(embeddingCount: textNumEmbeds + 1, dimensions: textDim)

        if convLayers > 0 {
            self.extraModeling = true
            self.freqsCis = precomputeFreqsCis(dim: textDim, end: precomputeMaxPos)
            self.text_blocks = Sequential(
                layers: (0 ..< convLayers).map { _ in F5ConvNeXtV2Block(dim: textDim, intermediateDim: textDim * convMult) }
            )
        }
    }

    func callAsFunction(_ inText: MLXArray, seqLen: Int, dropText: Bool = false) -> MLXArray {
        var text = inText + MLXArray([1])
        let batchSize = text.shape[0]
        let textLen = text.shape[1]

        if textLen > seqLen {
            text = text[0..., 0 ..< seqLen]
        }

        if textLen < seqLen {
            text = MLX.padded(text, widths: [.init((0, 0)), .init((0, seqLen - textLen))], value: MLXArray(0))
        }

        if dropText {
            text = MLX.zeros(like: text)
        }

        var output = text_embed(text)

        if extraModeling, let freqsCis = freqsCis, let textBlocks = text_blocks {
            let batchStart = MLX.zeros([batchSize], type: Int32.self)
            let posIdx = getPosEmbedIndices(start: batchStart, length: seqLen, maxPos: precomputeMaxPos)
            let textPosEmbed = freqsCis[posIdx]
            output = output + textPosEmbed
            output = textBlocks(output)
        }

        return output
    }
}
