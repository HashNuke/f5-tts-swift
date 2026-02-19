import Foundation
import MLX
import MLXNN
import MLXRandom

// feed forward

class F5FeedForward: Module {
    let ff: Sequential

    init(dim: Int, dimOut: Int? = nil, mult: Int = 4, dropout: Float = 0.0, approximate: String = "none") {
        let innerDim = Int(dim * mult)
        let outputDim = dimOut ?? dim

        let activation = GELU(approximation: approximate == "tanh" ? .tanh : .none)

        let projectIn = Sequential(layers: [
            Linear(dim, innerDim),
            activation
        ])

        self.ff = Sequential(layers: [
            projectIn,
            Dropout(p: dropout),
            Linear(innerDim, outputDim)
        ])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return ff(x)
    }
}
