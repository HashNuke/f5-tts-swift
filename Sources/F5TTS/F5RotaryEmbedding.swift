import Foundation
import MLX
import MLXFast
import MLXNN

class F5RotaryEmbedding: Module {
    let inv_freq: MLXArray
    let interpolationFactor: Float

    init(
        dim: Int,
        useXpos: Bool = false,
        scaleBase: Int = 512,
        interpolationFactor: Float = 1.0,
        base: Float = 10000.0,
        baseRescaleFactor: Float = 1.0
    ) {
        let adjustedBase = base * pow(baseRescaleFactor, Float(dim) / Float(dim - 2))
        self.inv_freq = 1.0 / pow(adjustedBase, MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim))

        assert(interpolationFactor >= 1.0, "Interpolation factor must be >= 1.0")
        self.interpolationFactor = interpolationFactor
    }

    func forwardFromSeqLen(_ seqLen: Int) -> (MLXArray, Float) {
        let t = MLXArray(0..<seqLen).asType(.float32)
        return callAsFunction(t)
    }

    func callAsFunction(_ t: MLXArray) -> (MLXArray, Float) {
        var freqs = MLX.matmul(t.expandedDimensions(axis: 1).asType(inv_freq.dtype), inv_freq.expandedDimensions(axis: 0))
        freqs = freqs / interpolationFactor

        freqs = MLX.stacked([freqs, freqs], axis: -1)
        let newShape = Array(
            freqs.shape.dropLast(2) +
                [freqs.shape[freqs.shape.count - 2] * freqs.shape[freqs.shape.count - 1]]
        )
        freqs = MLX.reshaped(freqs, newShape)
        return (freqs, 1.0)
    }
}
