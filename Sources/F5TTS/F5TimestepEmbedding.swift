import Foundation
import MLX
import MLXNN
import MLXRandom

// time step conditioning embedding

class F5TimestepEmbedding: Module {
    let time_embed: F5SinusPositionEmbedding
    let time_mlp: Sequential

    init(dim: Int, freqEmbedDim: Int = 256) {
        self.time_embed = F5SinusPositionEmbedding(dim: freqEmbedDim)

        self.time_mlp = Sequential(
            layers: [
                Linear(freqEmbedDim, dim),
                SiLU(),
                Linear(dim, dim),
            ]
        )
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let timeHidden = time_embed(timestep)
        let time = time_mlp(timeHidden)
        return time
    }
}
