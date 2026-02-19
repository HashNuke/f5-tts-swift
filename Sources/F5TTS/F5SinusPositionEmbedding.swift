import Foundation
import MLX
import MLXNN
import MLXRandom

// sinusoidal position embedding

class F5SinusPositionEmbedding: Module {
    let dim: Int

    init(dim: Int) {
        self.dim = dim
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let scale: Float = 1000.0
        let halfDim = dim / 2

        let emb = log(10000.0) / Float(halfDim - 1)
        let expEmb = MLX.exp(MLXArray(0..<halfDim) * -emb)

        let expandedX = MLX.expandedDimensions(inputs, axis: 1)
        let expandedEmb = MLX.expandedDimensions(expEmb, axis: 0)

        let scaled = scale * expandedX * expandedEmb
        let sinEmb = MLX.sin(scaled)
        let cosEmb = MLX.cos(scaled)

        let output = MLX.concatenated([sinEmb, cosEmb], axis: -1)
        return output
    }
}
