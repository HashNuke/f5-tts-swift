import Foundation
import MLX
import MLXFast
import MLXFFT
import MLXLinalg
import MLXNN
import MLXRandom

// rotary positional embedding related

func precomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000.0, thetaRescaleFactor: Float = 1.0) -> MLXArray {
    let range = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)[0..<(dim / 2)]
    let freqs = 1.0 / MLX.pow(MLXArray(theta), range / Float(dim))

    let t = MLXArray(0..<end).asType(.float32)
    let outerFreqs = MLX.outer(t, freqs).asType(.float32)

    let freqsCos = outerFreqs.cos()
    let freqsSin = outerFreqs.sin()

    return MLX.concatenated([freqsCos, freqsSin], axis: -1)
}

func getPosEmbedIndices(start: MLXArray, length: Int, maxPos: Int, scale: Float = 1.0) -> MLXArray {
    let scaleArray = MLX.ones(like: start).asType(.float32) * scale

    let pos = MLX.expandedDimensions(start, axis: 1) +
        (MLXArray(0..<length).expandedDimensions(axis: 0) * scaleArray.expandedDimensions(axis: 1)).asType(.int32)

    return MLX.where(pos .< maxPos, pos, maxPos - 1)
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    let newShape = Array(shape.dropLast() + [shape.last! / 2, 2])
    let reshapedX = x.reshaped(newShape)

    let x1x2 = reshapedX.split(parts: 2, axis: -1)
    let x1 = x1x2[0]
    let x2 = x1x2[1]

    let squeezedX1 = x1.squeezed(axis: -1)
    let squeezedX2 = x2.squeezed(axis: -1)

    let stackedX = MLX.stacked([-squeezedX2, squeezedX1], axis: -1)

    let finalShape = Array(stackedX.shape.dropLast(2) + [stackedX.shape[stackedX.shape.count - 2] * stackedX.shape[stackedX.shape.count - 1]])
    let result = stackedX.reshaped(finalShape)

    return result
}

func applyRotaryPosEmb(t: MLXArray, freqs: MLXArray, scale: Float = 1.0) -> MLXArray {
    let rotDim = freqs.shape[freqs.shape.count - 1]
    let seqLen = t.shape[t.shape.count - 2]

    let freqsTrimmed = freqs[(-seqLen)..., 0...]
    let scaleAdjusted = MLXArray(scale)

    var freqsRearranged = freqsTrimmed
    if t.ndim == 4 && freqsRearranged.ndim == 3 {
        freqsRearranged = freqsRearranged.reshaped([freqsRearranged.shape[0], 1, freqsRearranged.shape[1], freqsRearranged.shape[2]])
    }

    let tRotated = t[.ellipsis, 0..<rotDim]
    let tUnrotated = t[.ellipsis, rotDim..<t.shape[t.shape.count - 1]]
    let rotatedT = (tRotated * freqsRearranged.cos() * scaleAdjusted) +
        (rotateHalf(tRotated) * freqsRearranged.sin() * scaleAdjusted)
    let out = MLX.concatenated([rotatedT, tUnrotated], axis: -1)

    return out
}
