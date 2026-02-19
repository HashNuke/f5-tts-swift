import Foundation
import MLX
import MLXNN
import MLXRandom

public class F5DurationPredictor: Module {
    enum DurationPredictorError: Error {
        case unableToLoadModel
        case unableToLoadReferenceAudio
        case unableToDetermineDuration
    }

    public let melSpec: F5MelSpec
    public let transformer: F5DurationTransformer

    let dim: Int
    let numChannels: Int
    let vocabCharMap: [String: Int]
    let to_pred: Sequential

    init(
        transformer: F5DurationTransformer,
        melSpec: F5MelSpec,
        vocabCharMap: [String: Int]
    ) {
        self.melSpec = melSpec
        self.numChannels = self.melSpec.nMels
        self.transformer = transformer
        self.dim = transformer.dim
        self.vocabCharMap = vocabCharMap

        self.to_pred = Sequential(layers: [
            Linear(dim, 1, bias: false), Softplus()
        ])
    }

    func callAsFunction(_ cond: MLXArray, text: [String]) -> MLXArray {
        var cond = cond

        // raw wave

        if cond.ndim == 2 {
            cond = cond.reshaped([cond.shape[1]])
            cond = melSpec(x: cond)
        }

        let batch = cond.shape[0]
        let condSeqLen = cond.shape[1]
        var lens = MLX.full([batch], values: condSeqLen, type: Int.self)

        // text

        let inputText = listStrToIdx(text, vocabCharMap: vocabCharMap)
        let textLens = (inputText .!= -1).sum(axis: -1)
        lens = MLX.maximum(textLens, lens)

        var output = transformer(cond: cond, text: inputText)
        output = to_pred(output).mean().reshaped([batch, -1])
        output.eval()

        return output
    }
}
