import Foundation
import MLX
import MLXNN
import MLXRandom

class F5ConvPositionEmbedding: Module {
    let conv1d: Sequential

    init(dim: Int, kernelSize: Int = 31, groups: Int = 16) {
        precondition(kernelSize % 2 != 0, "Kernel size must be odd.")

        self.conv1d = Sequential(layers: [
            F5GroupableConv1d(inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: kernelSize / 2, groups: groups),
            Mish(),
            F5GroupableConv1d(inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, padding: kernelSize / 2, groups: groups),
            Mish()
        ])
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var input = x

        if let mask = mask {
            let expandedMask = MLX.expandedDimensions(mask, axis: -1)
            input = input * expandedMask
        }

        var output = conv1d(input)

        if let mask = mask {
            let expandedMask = MLX.expandedDimensions(mask, axis: -1)
            output = output * expandedMask
        }

        return output
    }
}
