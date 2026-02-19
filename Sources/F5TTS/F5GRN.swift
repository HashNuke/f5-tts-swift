// global response normalization
import Foundation
import MLX
import MLXNN

class GRN: Module {
    var gamma: MLXArray
    var beta: MLXArray

    init(dim: Int) {
        self.gamma = MLX.zeros([1, 1, dim])
        self.beta = MLX.zeros([1, 1, dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let Gx = MLXLinalg.norm(x, ord: 2, axis: 1, keepDims: true)
        let Nx = Gx / (Gx.mean(axis: -1, keepDims: true) + 1e-6)
        let output = gamma * (x * Nx) + beta + x
        return output
    }
}
